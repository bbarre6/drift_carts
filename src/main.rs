use avian3d::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::PrimitiveTopology;
use std::f32::consts::{FRAC_PI_2, PI};

/// Stadium-oval track: two straights joined by two 180° turns. Good for drifting
/// since every corner is a wide sustained arc rather than a sharp hairpin.
const HALF_STRAIGHT: f32 = 30.0;
const TURN_RADIUS: f32 = 22.0;
const TRACK_WIDTH: f32 = 12.0;
/// Points sampled per 180° turn; higher = smoother curve, more wall segments.
/// Bumped up so the (now bigger) turns don't look faceted/polygonal.
const ARC_STEPS: usize = 48;
/// Target length of one barrier/road chunk, in world units. Used to subdivide
/// the straights (previously a single huge segment each) and to pick each
/// segment's stripe color by cumulative distance rather than segment index --
/// without this, the turns (many short ~1.4-unit arc segments) flickered
/// through colors far faster than the straights (one segment, one color),
/// an inconsistent, messy-looking stripe cadence.
const STRIPE_LENGTH: f32 = 4.0;

/// Second level: a full copy of the oval above, raised up and shifted east far
/// enough that its footprint never overlaps the ground oval's.
const ELEVATED_HEIGHT: f32 = 8.0;
const ELEVATED_OFFSET_X: f32 = 110.0;
/// Where the connecting ramps start along their straight -- well clear of the
/// existing jump ramps (which sit near x=0) and far enough from the turn at
/// each straight's end that the car has climbed well past barrier height
/// (0.6) before it gets anywhere near that turn's geometry.
const CONNECTOR_RAMP_START_X: f32 = 10.0;
/// Reaches exactly ELEVATED_OFFSET_X - HALF_STRAIGHT (80.0): the elevated
/// oval's own straight start, so the ramp hands off flush with no seam.
const CONNECTOR_RAMP_LENGTH: f32 = ELEVATED_OFFSET_X - HALF_STRAIGHT - CONNECTOR_RAMP_START_X;

/// Debug grid: a numbered world-X marker line every DEBUG_MARKER_SPACING
/// units, spanning the full width (Z) of the map. Useful for reading off
/// exact world X coordinates while driving -- e.g. to check a ramp zone's
/// boundaries against where it actually sits on the ground.
const DEBUG_MARKER_SPACING: f32 = 10.0;
const DEBUG_MARKER_X_MIN: f32 = -60.0;
const DEBUG_MARKER_X_MAX: f32 = 200.0;
const DEBUG_MARKER_Z_HALF_EXTENT: f32 = 200.0;

/// Size of one seven-segment digit painted flat on the ground next to each
/// debug marker line (see `digit_segments`/`spawn_debug_number`).
const DIGIT_WIDTH: f32 = 0.6;
const DIGIT_HEIGHT: f32 = 1.2;
const DIGIT_SPACING: f32 = 0.2;
const DIGIT_SEGMENT_THICKNESS: f32 = 0.15;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, PhysicsPlugins::default()))
        .add_systems(Startup, setup)
        .init_resource::<DebugMode>()
        .add_systems(
            Update,
            (
                car_controller,
                ramp_zones,
                car_visual_tilt,
                drift_effects,
                update_particles,
                toggle_debug_mode,
                camera_follow,
            )
                .chain(),
        )
        .run();
}

/// Whether the debug coordinate-grid overlay (backquote key) is showing.
#[derive(Resource, Default)]
struct DebugMode(bool);

/// Tags every entity that belongs to the debug overlay (both the ground
/// grid lines and the number labels), so `toggle_debug_mode` can flip all
/// of their visibility at once with a single query.
#[derive(Component)]
struct DebugEntity;

/// The on-screen clickable button that toggles debug mode (an alternative
/// to the backquote key, for anyone who'd rather click than remember a key).
#[derive(Component)]
struct DebugToggleButton;

/// Tunable arcade handling constants for a drivable car.
#[derive(Component)]
struct Car {
    /// Forward/reverse acceleration.
    engine_power: f32,
    /// Hard cap on horizontal speed.
    max_speed: f32,
    /// Max yaw rate (rad/s) at full steering input and full speed.
    turn_speed: f32,
    /// `turn_speed` is multiplied by this while drifting, so the car rotates
    /// harder into the slide instead of just sliding straight.
    drift_turn_multiplier: f32,
    /// How fast sideways (lateral) velocity is cancelled out under normal grip.
    /// High = sticks to the road. Low = slides.
    grip: f32,
    /// Same as `grip`, but applied while the handbrake/drift key is held.
    drift_grip: f32,
}

impl Default for Car {
    fn default() -> Self {
        Self {
            engine_power: 34.0,
            max_speed: 33.0,
            turn_speed: 2.6,
            drift_turn_multiplier: 1.5,
            grip: 9.0,
            drift_grip: 1.1,
        }
    }
}

/// Marks the visual-only glTF model child of the car's physics body. Its local
/// rotation gets a cosmetic pitch tilt (see `car_visual_tilt`) based on climb/fall
/// rate -- purely visual, so it can never feed back into the physics body itself.
#[derive(Component)]
struct CarVisual;

/// Shared mesh/material handles for drift smoke and sparks, built once in `setup`
/// so every particle spawn just reuses the same handles instead of allocating new ones.
#[derive(Resource)]
struct DriftEffectAssets {
    smoke_mesh: Handle<Mesh>,
    smoke_material: Handle<StandardMaterial>,
    spark_mesh: Handle<Mesh>,
    spark_material: Handle<StandardMaterial>,
}

/// A short-lived visual-only effect (smoke puff or spark). Grows/shrinks from
/// `start_scale` to `end_scale` over `lifetime` seconds, then despawns.
#[derive(Component)]
struct Particle {
    velocity: Vec3,
    age: f32,
    lifetime: f32,
    start_scale: f32,
    end_scale: f32,
    gravity: f32,
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground: visual plane + a thin static collider under it. Sized to cover
    // both the ground oval AND the elevated oval's full east-shifted footprint
    // (out to ELEVATED_OFFSET_X + half_straight + turn_radius), with margin,
    // so there's grass run-off everywhere, including under the second level.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(420.0, 420.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.5, 0.3),
            ..default()
        })),
    ));
    commands.spawn((
        RigidBody::Static,
        Collider::cuboid(420.0, 0.2, 420.0),
        Transform::from_xyz(0.0, -0.1, 0.0),
    ));

    // Ground-level oval: the original track, unchanged, red/white barriers.
    spawn_track(
        &mut commands,
        &mut meshes,
        &mut materials,
        HALF_STRAIGHT,
        TURN_RADIUS,
        TRACK_WIDTH,
        Vec3::ZERO,
        Color::srgb(0.8, 0.1, 0.1),
        Color::srgb(0.9, 0.9, 0.85),
    );

    // Elevated oval: an identical second loop, shifted east and raised up,
    // with blue/white barriers so it reads as a distinct "second level."
    spawn_track(
        &mut commands,
        &mut meshes,
        &mut materials,
        HALF_STRAIGHT,
        TURN_RADIUS,
        TRACK_WIDTH,
        Vec3::new(ELEVATED_OFFSET_X, ELEVATED_HEIGHT, 0.0),
        Color::srgb(0.15, 0.35, 0.85),
        Color::srgb(0.9, 0.9, 0.9),
    );

    // Elevated oval's physical floor: a full solid platform under its whole
    // footprint (road, turns, and infield alike) -- the road mesh itself has
    // no collider of its own (same as ground level), so without this the car
    // would have nothing to actually rest on once it reaches the second level.
    //
    // Both half-extents need the + TRACK_WIDTH/2.0 padding: the track's
    // CENTERLINE spans only HALF_STRAIGHT+TURN_RADIUS, but the actual road
    // (and its outer barrier) sticks out another TRACK_WIDTH/2 past that at
    // a turn's apex -- missing this on the X side left the floor's edge
    // short of the real outer barrier near a turn, so driving the outside
    // line there drove off the edge of the solid floor into open air.
    let elevated_floor_half_x = HALF_STRAIGHT + TURN_RADIUS + TRACK_WIDTH / 2.0;
    let elevated_floor_half_z = TURN_RADIUS + TRACK_WIDTH / 2.0;
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(
            elevated_floor_half_x * 2.0,
            0.08,
            elevated_floor_half_z * 2.0,
        ))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.5, 0.52),
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::from_xyz(ELEVATED_OFFSET_X, ELEVATED_HEIGHT, 0.0),
    ));
    commands.spawn((
        RigidBody::Static,
        Collider::cuboid(
            elevated_floor_half_x * 2.0,
            0.2,
            elevated_floor_half_z * 2.0,
        ),
        Transform::from_xyz(ELEVATED_OFFSET_X, ELEVATED_HEIGHT - 0.1, 0.0),
    ));

    // Support pillars: purely visual (and a collider, so nothing can clip through
    // one by accident), standing the platform on something instead of leaving it
    // looking like it's just floating in midair. Three evenly spaced under each
    // straight, both sides of the oval.
    let pillar_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.45, 0.45, 0.47),
        perceptual_roughness: 0.85,
        ..default()
    });
    let pillar_mesh = meshes.add(Cuboid::new(1.2, ELEVATED_HEIGHT, 1.2));
    for straight_z in [-TURN_RADIUS, TURN_RADIUS] {
        for t in [0.2_f32, 0.5, 0.8] {
            let x = ELEVATED_OFFSET_X - HALF_STRAIGHT + t * (2.0 * HALF_STRAIGHT);
            commands.spawn((
                RigidBody::Static,
                Collider::cuboid(1.2, ELEVATED_HEIGHT, 1.2),
                Mesh3d(pillar_mesh.clone()),
                MeshMaterial3d(pillar_material.clone()),
                Transform::from_xyz(x, ELEVATED_HEIGHT / 2.0, straight_z),
            ));
        }
    }

    // Soft fill light + a proper sky color, instead of the default near-black
    // void and harsh shadow-only lighting.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.75, 0.8, 1.0),
        brightness: 400.0,
        ..default()
    });
    commands.insert_resource(ClearColor(Color::srgb(0.55, 0.75, 0.95)));

    // Two long, gentle connector ramps -- real drivable lanes between the two
    // levels, not jumps -- one per straight, both climbing toward +X to reach
    // the elevated oval's straight-start on that same side.
    spawn_ramp(
        &mut commands,
        &mut meshes,
        &mut materials,
        CONNECTOR_RAMP_START_X,
        TURN_RADIUS,
        1.0,
        CONNECTOR_RAMP_LENGTH,
        ELEVATED_HEIGHT,
        TRACK_WIDTH,
        Color::srgb(0.35, 0.35, 0.38),
        false,
    );
    spawn_ramp(
        &mut commands,
        &mut meshes,
        &mut materials,
        CONNECTOR_RAMP_START_X,
        -TURN_RADIUS,
        1.0,
        CONNECTOR_RAMP_LENGTH,
        ELEVATED_HEIGHT,
        TRACK_WIDTH,
        Color::srgb(0.35, 0.35, 0.38),
        false,
    );

    // One jump ramp per straight, narrower than the road so there's room to drive
    // around on either side if you'd rather not catch air. Directions match the
    // car's start-facing below (now reversed, so the loop is driven the other way).
    spawn_ramp(
        &mut commands,
        &mut meshes,
        &mut materials,
        1.0,
        -TURN_RADIUS,
        -1.0,
        5.0,
        1.8,
        5.0,
        Color::srgb(0.55, 0.3, 0.15),
        true,
    );
    spawn_ramp(
        &mut commands,
        &mut meshes,
        &mut materials,
        -1.0,
        TURN_RADIUS,
        1.0,
        5.0,
        1.8,
        5.0,
        Color::srgb(0.55, 0.3, 0.15),
        true,
    );

    commands.insert_resource(DriftEffectAssets {
        smoke_mesh: meshes.add(Sphere::new(1.0)),
        smoke_material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.6, 0.6),
            perceptual_roughness: 1.0,
            ..default()
        }),
        spark_mesh: meshes.add(Sphere::new(1.0)),
        spark_material: materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.7, 0.2),
            emissive: LinearRgba::rgb(6.0, 2.5, 0.2),
            ..default()
        }),
    });

    // Car physics body (invisible box) with the Kenney model as a visual-only child.
    // Starts at the opposite (east) end of the bottom straight, facing -X, so the
    // loop is driven the other way around. Drops in from y=1 so it settles onto
    // the road on its own.
    let start_pos = Vec3::new(HALF_STRAIGHT, 1.0, -TURN_RADIUS);
    let car_body = commands
        .spawn((
            Car::default(),
            RigidBody::Dynamic,
            Collider::cuboid(1.0, 0.6, 2.0),
            LinearVelocity::ZERO,
            AngularVelocity::ZERO,
            // Lock pitch and roll: letting the physics body itself tip up a ramp
            // (or pitch from clipping a wall while drifting) repeatedly turned
            // into the car being launched off the map -- any tilt redirects
            // engine thrust, and a collision at an angle can impart a lot of it.
            // The ramp tilt is now a purely cosmetic effect on the visual model
            // (see car_visual_tilt) that can't feed back into the physics body.
            LockedAxes::new().lock_rotation_x().lock_rotation_z(),
            // Safety cap: nothing in normal play should ever need to spin this
            // fast; this just stops a runaway spin from becoming an instant,
            // absurd launch.
            MaxAngularSpeed(6.0),
            Transform::from_translation(start_pos).looking_to(Vec3::NEG_X, Vec3::Y),
            Visibility::default(),
        ))
        .id();

    commands.entity(car_body).with_children(|parent| {
        parent.spawn((
            CarVisual,
            WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/race.glb"))),
            Transform::from_xyz(0.0, -0.3, 0.0),
        ));
    });

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Chase camera; camera_follow() drives it every frame.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.2, -4.0).looking_at(Vec3::new(0.0, 0.8, 0.0), Vec3::Y),
    ));

    // Debug coordinate grid (hidden by default; toggled with the ` key by
    // toggle_debug_mode). One bright line + one number label per world-X
    // marker, spanning the full width of the map -- spawned once at ground
    // height and again at the elevated level's height, so the grid reads
    // correctly no matter which level you're currently driving on.
    let debug_line_mesh = meshes.add(Cuboid::new(0.3, 0.05, DEBUG_MARKER_Z_HALF_EXTENT * 2.0));
    let debug_line_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.1, 1.0),
        emissive: LinearRgba::rgb(1.5, 0.1, 1.5),
        unlit: true,
        ..default()
    });
    // Digit segment meshes, shared by every number painted on the ground --
    // every segment of every digit is one of just these two box shapes.
    let digit_h_mesh = meshes.add(Cuboid::new(DIGIT_WIDTH, 0.03, DIGIT_SEGMENT_THICKNESS));
    let digit_v_mesh = meshes.add(Cuboid::new(
        DIGIT_SEGMENT_THICKNESS,
        0.03,
        DIGIT_HEIGHT / 2.0,
    ));

    for level_y in [0.0, ELEVATED_HEIGHT] {
        let mut marker_x = DEBUG_MARKER_X_MIN;
        while marker_x <= DEBUG_MARKER_X_MAX {
            commands.spawn((
                DebugEntity,
                Visibility::Hidden,
                Mesh3d(debug_line_mesh.clone()),
                MeshMaterial3d(debug_line_material.clone()),
                Transform::from_xyz(marker_x, level_y + 0.12, 0.0),
            ));
            // Offset perpendicular to the line -- the line runs lengthwise
            // along Z (spanning the whole map), so moving along Z just slides
            // along the same line; X is the direction that actually steps off
            // of it.
            const DEBUG_NUMBER_X_OFFSET: f32 = 1.5;
            // One copy of the number on each straight (z = +/-TURN_RADIUS)
            // instead of a single copy at z=0 -- the track's straights sit
            // out at +/-TURN_RADIUS, not through the middle of the map, so a
            // single centered copy was stranded in the empty infield for
            // every marker that falls within a straight's X range.
            for track_z in [-TURN_RADIUS, TURN_RADIUS] {
                spawn_debug_number(
                    &mut commands,
                    &digit_h_mesh,
                    &digit_v_mesh,
                    &debug_line_material,
                    &format!("{marker_x:.0}"),
                    Vec3::new(marker_x + DEBUG_NUMBER_X_OFFSET, level_y + 0.19, track_z),
                );
            }
            marker_x += DEBUG_MARKER_SPACING;
        }
    }

    // On-screen clickable button that does the same thing as pressing `.
    commands
        .spawn((
            DebugToggleButton,
            Button,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(10.0),
                top: Val::Px(10.0),
                width: Val::Px(120.0),
                height: Val::Px(40.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.15, 0.15, 0.18)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("DEBUG"),
                TextFont {
                    font_size: 16.0.into(),
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
        });
}

/// Returns the center (local x, z) and orientation (true = horizontal bar,
/// false = vertical bar) of each lit segment for one character, using a
/// classic seven-segment layout. Local space: x right, z up, origin at the
/// character's bottom-left corner, sized DIGIT_WIDTH x DIGIT_HEIGHT.
fn digit_segments(c: char) -> &'static [(f32, f32, bool)] {
    const W: f32 = DIGIT_WIDTH;
    const H: f32 = DIGIT_HEIGHT;
    const TOP: (f32, f32, bool) = (W / 2.0, H, true);
    const MIDDLE: (f32, f32, bool) = (W / 2.0, H / 2.0, true);
    const BOTTOM: (f32, f32, bool) = (W / 2.0, 0.0, true);
    const TOP_LEFT: (f32, f32, bool) = (0.0, H * 0.75, false);
    const TOP_RIGHT: (f32, f32, bool) = (W, H * 0.75, false);
    const BOTTOM_LEFT: (f32, f32, bool) = (0.0, H * 0.25, false);
    const BOTTOM_RIGHT: (f32, f32, bool) = (W, H * 0.25, false);

    match c {
        '0' => &[TOP, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, BOTTOM],
        '1' => &[TOP_RIGHT, BOTTOM_RIGHT],
        '2' => &[TOP, TOP_RIGHT, MIDDLE, BOTTOM_LEFT, BOTTOM],
        '3' => &[TOP, TOP_RIGHT, MIDDLE, BOTTOM_RIGHT, BOTTOM],
        '4' => &[TOP_LEFT, TOP_RIGHT, MIDDLE, BOTTOM_RIGHT],
        '5' => &[TOP, TOP_LEFT, MIDDLE, BOTTOM_RIGHT, BOTTOM],
        '6' => &[TOP, TOP_LEFT, MIDDLE, BOTTOM_LEFT, BOTTOM_RIGHT, BOTTOM],
        '7' => &[TOP, TOP_RIGHT, BOTTOM_RIGHT],
        '8' => &[TOP, TOP_LEFT, TOP_RIGHT, MIDDLE, BOTTOM_LEFT, BOTTOM_RIGHT, BOTTOM],
        '9' => &[TOP, TOP_LEFT, TOP_RIGHT, MIDDLE, BOTTOM_RIGHT],
        '-' => &[MIDDLE],
        _ => &[],
    }
}

/// Paints a number flat on the ground out of seven-segment-digit meshes
/// (see `digit_segments`), centered horizontally and vertically on `center`.
/// All spawned entities are tagged `DebugEntity` and start hidden, exactly
/// like the grid lines, so `toggle_debug_mode` shows/hides them together.
fn spawn_debug_number(
    commands: &mut Commands,
    h_mesh: &Handle<Mesh>,
    v_mesh: &Handle<Mesh>,
    material: &Handle<StandardMaterial>,
    text: &str,
    center: Vec3,
) {
    let char_count = text.chars().count() as f32;
    let total_width = char_count * DIGIT_WIDTH + (char_count - 1.0).max(0.0) * DIGIT_SPACING;
    let start_x = center.x - total_width / 2.0;

    for (i, c) in text.chars().enumerate() {
        let digit_origin = Vec3::new(
            start_x + i as f32 * (DIGIT_WIDTH + DIGIT_SPACING),
            center.y,
            center.z - DIGIT_HEIGHT / 2.0,
        );
        for &(local_x, local_z, horizontal) in digit_segments(c) {
            let mesh = if horizontal { h_mesh.clone() } else { v_mesh.clone() };
            // Mirrored vertically (top<->bottom) -- without this the digits
            // read upside down from the usual driving/viewing angle.
            let mirrored_z = DIGIT_HEIGHT - local_z;
            commands.spawn((
                DebugEntity,
                Visibility::Hidden,
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                Transform::from_translation(digit_origin + Vec3::new(local_x, 0.0, mirrored_z)),
            ));
        }
    }
}

/// Centerline of an oval of the given dimensions, in traversal order, relative
/// to its own local origin: bottom straight (west→east) -> east 180° turn ->
/// top straight (east→west) -> west 180° turn -> back to start. The loop
/// closes implicitly (last point connects back to points[0]).
///
/// Each straight is chopped into ~STRIPE_LENGTH-long pieces (previously a
/// single segment spanning its whole length) purely so `spawn_track` has
/// enough segments along it to stripe -- the shape is still perfectly
/// straight, just represented as several collinear points instead of two.
fn track_centerline(half_straight: f32, turn_radius: f32) -> Vec<Vec3> {
    let mut points = Vec::new();

    // Bottom straight: west -> east.
    let straight_len = 2.0 * half_straight;
    let straight_steps = ((straight_len / STRIPE_LENGTH).round() as usize).max(1);
    for i in 0..straight_steps {
        let t = i as f32 / straight_steps as f32;
        points.push(Vec3::new(-half_straight + straight_len * t, 0.0, -turn_radius));
    }

    // East 180° turn: bottom straight's end -> top straight's start.
    let east_center = Vec3::new(half_straight, 0.0, 0.0);
    for i in 0..ARC_STEPS {
        let theta = -FRAC_PI_2 + (i as f32 / ARC_STEPS as f32) * PI;
        points.push(east_center + Vec3::new(turn_radius * theta.cos(), 0.0, turn_radius * theta.sin()));
    }

    // Top straight: east -> west.
    for i in 0..straight_steps {
        let t = i as f32 / straight_steps as f32;
        points.push(Vec3::new(half_straight - straight_len * t, 0.0, turn_radius));
    }

    // West 180° turn: top straight's end -> bottom straight's start.
    let west_center = Vec3::new(-half_straight, 0.0, 0.0);
    for i in 0..ARC_STEPS {
        let theta = FRAC_PI_2 + (i as f32 / ARC_STEPS as f32) * PI;
        points.push(west_center + Vec3::new(turn_radius * theta.cos(), 0.0, turn_radius * theta.sin()));
    }

    points
}

/// Builds an oval track by stitching a box ("road") plus two collidable barrier
/// walls onto every segment of the centerline loop, translated by `offset` --
/// used for both the ground oval (offset zero) and the elevated one.
fn spawn_track(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    half_straight: f32,
    turn_radius: f32,
    track_width: f32,
    offset: Vec3,
    barrier_color_a: Color,
    barrier_color_b: Color,
) {
    let points: Vec<Vec3> = track_centerline(half_straight, turn_radius)
        .into_iter()
        .map(|p| p + offset)
        .collect();
    let count = points.len();

    let road_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.1, 0.12),
        perceptual_roughness: 0.95,
        ..default()
    });
    let lane_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.92, 0.88),
        perceptual_roughness: 0.7,
        ..default()
    });
    let barrier_materials = [
        materials.add(StandardMaterial {
            base_color: barrier_color_a,
            ..default()
        }),
        materials.add(StandardMaterial {
            base_color: barrier_color_b,
            ..default()
        }),
    ];

    // Cumulative distance along the loop, used (instead of segment index) to pick
    // each segment's stripe color -- gives a consistent stripe length everywhere,
    // including through the turns' many short arc segments (see STRIPE_LENGTH).
    let mut distance = 0.0_f32;

    for i in 0..count {
        let start = points[i];
        let end = points[(i + 1) % count];
        let segment = end - start;
        let length = segment.length();
        if length < 0.001 {
            continue;
        }

        let tangent = segment / length;
        let right = tangent.cross(Vec3::Y).normalize();
        let midpoint = (start + end) * 0.5;
        let rotation = Transform::from_translation(midpoint)
            .looking_to(tangent, Vec3::Y)
            .rotation;

        // Road surface, sitting slightly above the grass to avoid z-fighting.
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(track_width, 0.08, length))),
            MeshMaterial3d(road_material.clone()),
            Transform::from_translation(midpoint + Vec3::Y * 0.04).with_rotation(rotation),
        ));

        // Center lane line, just above the road surface.
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.25, 0.01, length))),
            MeshMaterial3d(lane_material.clone()),
            Transform::from_translation(midpoint + Vec3::Y * 0.085).with_rotation(rotation),
        ));

        // Collidable barrier walls on both edges, alternating color by distance
        // along the loop so stripes read as a consistent-length rumble strip.
        let stripe_index = (distance / STRIPE_LENGTH).floor() as i64;
        let barrier_material = barrier_materials[(stripe_index.rem_euclid(2)) as usize].clone();
        distance += length;
        for side in [-1.0_f32, 1.0] {
            let wall_pos = midpoint + right * side * (track_width / 2.0 + 0.15) + Vec3::Y * 0.3;
            commands.spawn((
                RigidBody::Static,
                Collider::cuboid(0.3, 0.6, length),
                Mesh3d(meshes.add(Cuboid::new(0.3, 0.6, length))),
                MeshMaterial3d(barrier_material.clone()),
                Transform::from_translation(wall_pos).with_rotation(rotation),
            ));
        }
    }
}

/// A ramp's footprint and geometry, used by `ramp_zones` to lift the car along
/// its slope kinematically (see that system's doc comment for why).
#[derive(Component)]
struct RampZone {
    /// World position of the ramp's entry edge, at ground height.
    origin: Vec3,
    /// Maps world space into the ramp's local space (local +X = up the slope).
    inverse_rotation: Quat,
    length: f32,
    height: f32,
    half_width: f32,
    /// true for a launch-into-the-air jump ramp (gets a velocity boost on
    /// exit); false for a connector ramp joining two drivable levels, which
    /// should just smoothly become flat ground on the other side, no launch.
    is_jump: bool,
    /// Was the car inside this zone as of the last check? Used to fire the
    /// launch boost exactly once, right as it exits over the far edge.
    car_was_inside: bool,
}

/// Builds a single jump ramp on one of the straights, as a visual-only wedge
/// (triangular prism) mesh paired with a `RampZone` -- there's no physics
/// collider on it at all.
///
/// Why: the car's rigid body can't pitch (that's locked, for overall driving
/// stability -- see the comment on LockedAxes in `setup`), so a real collider
/// here just means its flat underside plows into the slope instead of rolling
/// up it. That's both the steady loss of speed on approach and the janky,
/// clipping look climbing it. `ramp_zones` instead directly lifts the car's
/// height to match the ramp's profile while it's within the footprint, with no
/// collision involved, and gives it a clean upward velocity boost on exit.
///
/// `base_x` is the entry point (ground height); `track_z` is the straight's
/// centerline Z; `direction` is `+1.0`/`-1.0` for which way along X the car needs to
/// be driving to go up it (the two straights run opposite ways). `length`/`height`/
/// `width` size the wedge; `color` is its material; `is_jump` controls whether
/// `ramp_zones` fires a launch boost on exit (a short, steep jump ramp) or just
/// lets the car roll off onto flat ground at the new height (a long, gentle
/// connector ramp between levels).
fn spawn_ramp(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    base_x: f32,
    track_z: f32,
    direction: f32,
    length: f32,
    height: f32,
    width: f32,
    color: Color,
    is_jump: bool,
) {
    // Flush with the flat ground collider's top surface (see the static ground
    // collider in `setup`, at y=-0.1 with 0.2 thickness), which is what the car is
    // actually resting on -- not the cosmetic road mesh, which has no collider of
    // its own. Matching this instead of the road's visual height is what removes
    // the remaining step at the entry.
    const GROUND_SURFACE_Y: f32 = 0.0;

    let half_width = width / 2.0;
    // Local space: +X runs from the knife edge (x=0) to the raised back (x=length).
    let front_left = Vec3::new(0.0, 0.0, -half_width);
    let front_right = Vec3::new(0.0, 0.0, half_width);
    let back_left = Vec3::new(length, 0.0, -half_width);
    let back_right = Vec3::new(length, 0.0, half_width);
    let top_left = Vec3::new(length, height, -half_width);
    let top_right = Vec3::new(length, height, half_width);

    // Outward-wound triangles: bottom, sloped top, back wall, two side walls.
    let triangles = [
        [front_left, back_left, back_right],
        [front_left, back_right, front_right],
        [front_left, front_right, top_right],
        [front_left, top_right, top_left],
        [back_left, top_right, back_right],
        [back_left, top_left, top_right],
        [front_left, top_left, back_left],
        [front_right, back_right, top_right],
    ];
    let positions: Vec<[f32; 3]> = triangles
        .iter()
        .flatten()
        .map(|v| v.to_array())
        .collect();

    let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_computed_flat_normals();

    let rotation = if direction > 0.0 {
        Quat::IDENTITY
    } else {
        Quat::from_rotation_y(PI)
    };
    let origin = Vec3::new(base_x, GROUND_SURFACE_Y, track_z);

    let ramp_material = materials.add(StandardMaterial {
        base_color: color,
        perceptual_roughness: 0.9,
        ..default()
    });

    commands.spawn((
        RampZone {
            origin,
            inverse_rotation: rotation.inverse(),
            length,
            height,
            half_width,
            is_jump,
            car_was_inside: false,
        },
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(ramp_material),
        Transform::from_translation(origin).with_rotation(rotation),
    ));

    // Connector ramps (not jump ramps) get real guardrails along both sides --
    // tall enough to block the car at any point along the climb, regardless of
    // how high it's currently risen, since they're plain vertical walls rather
    // than ones sloped to match the ramp (much simpler, and just as effective:
    // a wall spanning the full climb height always overlaps the car's current
    // height somewhere along it). Without these, drifting off the side mid-climb
    // meant falling straight through with no support at all.
    if !is_jump {
        const GUARDRAIL_HEIGHT: f32 = 12.0;
        let guardrail_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.22, 0.22, 0.25),
            perceptual_roughness: 0.8,
            ..default()
        });
        for side in [-1.0_f32, 1.0] {
            let local_pos = Vec3::new(
                length / 2.0,
                GUARDRAIL_HEIGHT / 2.0,
                side * (half_width + 0.15),
            );
            let world_pos = origin + rotation * local_pos;
            commands.spawn((
                RigidBody::Static,
                Collider::cuboid(length, GUARDRAIL_HEIGHT, 0.3),
                Mesh3d(meshes.add(Cuboid::new(length, GUARDRAIL_HEIGHT, 0.3))),
                MeshMaterial3d(guardrail_material.clone()),
                Transform::from_translation(world_pos).with_rotation(rotation),
            ));
        }
    }
}

/// Lifts the car's height to match a ramp's slope while it's within the ramp's
/// footprint -- no collision, just directly following the known profile -- and
/// gives it a clean upward velocity boost the moment it crosses the far edge.
///
/// See the comment on `spawn_ramp` for why this isn't done via a real collider.
fn ramp_zones(
    mut ramp_query: Query<&mut RampZone>,
    mut car_query: Query<(&mut Transform, &mut LinearVelocity), With<Car>>,
) {
    // Half the car collider's height (see Collider::cuboid(1.0, 0.6, 2.0) in
    // setup) -- how far above the ramp's surface the car's center should sit,
    // matching how it normally rests on flat ground.
    const CAR_GROUND_CLEARANCE: f32 = 0.3;

    let Ok((mut transform, mut lin_vel)) = car_query.single_mut() else {
        return;
    };

    for mut ramp in &mut ramp_query {
        let local = ramp.inverse_rotation * (transform.translation - ramp.origin);
        let inside = local.x >= 0.0 && local.x <= ramp.length && local.z.abs() <= ramp.half_width;

        if inside {
            let climb = (local.x / ramp.length).clamp(0.0, 1.0);
            transform.translation.y = ramp.origin.y + climb * ramp.height + CAR_GROUND_CLEARANCE;
            // Riding the known slope, not falling -- let gravity resume once it exits.
            lin_vel.0.y = 0.0;
        } else if ramp.car_was_inside && ramp.is_jump {
            // Just crossed the far edge of a JUMP ramp: launch, trading none of
            // the horizontal speed that was preserved while climbing -- it was
            // the ongoing loss of that speed to the ramp's own collision drag
            // that this whole approach exists to avoid.
            //
            // A connector ramp (is_jump == false) skips this entirely: it already
            // ends exactly at the new level's resting height (see CAR_GROUND_CLEARANCE
            // below), so the car just rolls off onto flat ground with no launch.
            //
            // The raw version of this (speed * sin(ramp angle), no scaling) jumped
            // much higher once top speed was raised, since it scales directly with
            // how fast the car is going. LAUNCH_BOOST_SCALE tones that down to a
            // smaller hop regardless of how fast top speed gets tuned later, and
            // MAX_LAUNCH_SPEED caps it outright so it can't scale away again.
            const LAUNCH_BOOST_SCALE: f32 = 0.35;
            const MAX_LAUNCH_SPEED: f32 = 6.0;
            let horizontal_speed = Vec2::new(lin_vel.0.x, lin_vel.0.z).length();
            let launch_angle = ramp.height.atan2(ramp.length);
            let boost = horizontal_speed * launch_angle.sin() * LAUNCH_BOOST_SCALE;
            lin_vel.0.y += boost.min(MAX_LAUNCH_SPEED);
        }

        ramp.car_was_inside = inside;
    }
}

/// WASD / arrow keys to drive. Space is the dedicated drift button (drops grip so the
/// back end slides).
fn car_controller(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<(&Car, &Transform, &mut LinearVelocity, &mut AngularVelocity)>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    for (car, transform, mut lin_vel, mut ang_vel) in &mut query {
        let forward: Vec3 = transform.forward().into();
        let right: Vec3 = transform.right().into();

        // Inverted: W/Up now reverses, S/Down now accelerates forward; A/Left steers
        // right, D/Right steers left.
        let throttle = if keys.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]) {
            -1.0
        } else if keys.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]) {
            1.0
        } else {
            0.0
        };

        let steer = if keys.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]) {
            -1.0
        } else if keys.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]) {
            1.0
        } else {
            0.0
        };

        // Dedicated drift button: Space, and nothing else.
        let drifting = keys.pressed(KeyCode::Space);

        // Engine: push along the car's forward axis.
        let mut velocity = lin_vel.0 + forward * throttle * car.engine_power * dt;

        // Steering: yaw rate scales with speed so the car doesn't spin in place when stopped,
        // and reverses sign when reversing so it still steers the way you'd expect.
        let forward_speed = velocity.dot(forward);
        let speed_factor = (forward_speed.abs() / car.max_speed).clamp(0.0, 1.0);
        let direction_sign = if forward_speed.abs() < 0.2 {
            0.0
        } else {
            forward_speed.signum()
        };
        // Turn in harder while drifting, so the slide actually rotates the car
        // instead of just carrying it sideways in a straight line.
        let turn_speed = if drifting {
            car.turn_speed * car.drift_turn_multiplier
        } else {
            car.turn_speed
        };
        ang_vel.0 = Vec3::Y * (steer * turn_speed * (0.3 + 0.7 * speed_factor) * direction_sign);

        // Grip: decompose velocity into forward/lateral components relative to the car body,
        // then bleed off the lateral component toward zero. Normal grip kills it almost
        // instantly (no sliding); drift grip kills it slowly, so the car slides through the turn.
        let lateral_speed = velocity.dot(right);
        let forward_component = forward * velocity.dot(forward);
        let lateral_component = right * lateral_speed;

        let grip = if drifting { car.drift_grip } else { car.grip };
        let decay = (grip * dt).clamp(0.0, 1.0);
        let reduced_lateral = lateral_component * (1.0 - decay);

        velocity = forward_component + reduced_lateral + Vec3::Y * velocity.y;

        // Clamp horizontal speed only; let gravity keep doing its thing on Y.
        let horizontal = Vec3::new(velocity.x, 0.0, velocity.z);
        if horizontal.length() > car.max_speed {
            let clamped = horizontal.normalize() * car.max_speed;
            velocity.x = clamped.x;
            velocity.z = clamped.z;
        }

        lin_vel.0 = velocity;
    }
}

/// Cosmetic-only pitch tilt on the car's visual model: nose tilts up while
/// climbing (e.g. off a ramp), down while falling. Purely a rotation on the
/// glTF child's own local Transform -- the physics body (and its collider)
/// never moves, so this cannot destabilize the simulation the way physically
/// unlocking the body's own pitch repeatedly did.
fn car_visual_tilt(
    time: Res<Time>,
    car_query: Query<&LinearVelocity, With<Car>>,
    mut visual_query: Query<&mut Transform, With<CarVisual>>,
) {
    let Ok(velocity) = car_query.single() else {
        return;
    };
    let Ok(mut visual_transform) = visual_query.single_mut() else {
        return;
    };

    let horizontal_speed = Vec2::new(velocity.0.x, velocity.0.z).length();
    let tilt = velocity.0.y.atan2(horizontal_speed.max(1.0)).clamp(-0.5, 0.5);
    let target = Quat::from_rotation_x(tilt);

    let dt = time.delta_secs();
    visual_transform.rotation = visual_transform
        .rotation
        .slerp(target, (10.0 * dt).clamp(0.0, 1.0));
}

/// Spawns tire smoke (steady stream) and sparks (occasional bright bursts) at both
/// rear wheels whenever the drift button is held and the car is actually moving.
fn drift_effects(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    assets: Res<DriftEffectAssets>,
    mut spawn_cooldown: Local<f32>,
    car_query: Query<(&Transform, &LinearVelocity), With<Car>>,
) {
    let dt = time.delta_secs();
    *spawn_cooldown -= dt;

    let Ok((transform, lin_vel)) = car_query.single() else {
        return;
    };

    let speed = lin_vel.0.length();
    if !keys.pressed(KeyCode::Space) || speed < 3.0 {
        return;
    }
    if *spawn_cooldown > 0.0 {
        return;
    }
    *spawn_cooldown = 0.03;

    let forward: Vec3 = transform.forward().into();
    let back: Vec3 = transform.back().into();
    let right: Vec3 = transform.right().into();
    let rear_center = transform.translation + back * 0.9 + Vec3::Y * -0.2;

    for side in [-1.0_f32, 1.0] {
        let wheel_pos = rear_center + right * side * 0.45;

        // Tire smoke: drifts up and slightly outward/backward, grows as it dissipates.
        let smoke_velocity = back * rand::random_range(0.3..1.0)
            + right * side * rand::random_range(0.2..0.6)
            + Vec3::Y * rand::random_range(0.4..1.0);
        commands.spawn((
            Particle {
                velocity: smoke_velocity,
                age: 0.0,
                lifetime: rand::random_range(0.5..0.9),
                start_scale: 0.1,
                end_scale: rand::random_range(0.5..0.7),
                gravity: 0.0,
            },
            Mesh3d(assets.smoke_mesh.clone()),
            MeshMaterial3d(assets.smoke_material.clone()),
            Transform::from_translation(wheel_pos).with_scale(Vec3::splat(0.1)),
        ));

        // Sparks: rarer, fast, bright, short-lived, arc downward under gravity.
        if rand::random_range(0.0..1.0) < 0.35 {
            let spark_velocity = back * rand::random_range(1.0..2.5)
                + right * side * rand::random_range(0.5..1.5)
                + forward * rand::random_range(0.0..0.5)
                + Vec3::Y * rand::random_range(0.5..1.5);
            commands.spawn((
                Particle {
                    velocity: spark_velocity,
                    age: 0.0,
                    lifetime: rand::random_range(0.2..0.4),
                    start_scale: 0.08,
                    end_scale: 0.0,
                    gravity: 9.0,
                },
                Mesh3d(assets.spark_mesh.clone()),
                MeshMaterial3d(assets.spark_material.clone()),
                Transform::from_translation(wheel_pos).with_scale(Vec3::splat(0.08)),
            ));
        }
    }
}

/// Ages and moves every active particle, despawning it once its lifetime expires.
fn update_particles(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut Particle, &mut Transform)>,
) {
    let dt = time.delta_secs();
    for (entity, mut particle, mut transform) in &mut query {
        particle.age += dt;
        if particle.age >= particle.lifetime {
            commands.entity(entity).despawn();
            continue;
        }

        particle.velocity.y -= particle.gravity * dt;
        transform.translation += particle.velocity * dt;

        let t = particle.age / particle.lifetime;
        let scale = particle.start_scale + (particle.end_scale - particle.start_scale) * t;
        transform.scale = Vec3::splat(scale.max(0.0));
    }
}

fn camera_follow(
    car_query: Query<&Transform, With<Car>>,
    mut camera_query: Query<&mut Transform, (With<Camera3d>, Without<Car>)>,
    time: Res<Time>,
) {
    let Ok(car_transform) = car_query.single() else {
        return;
    };
    let Ok(mut camera_transform) = camera_query.single_mut() else {
        return;
    };

    let back: Vec3 = car_transform.back().into();
    let forward: Vec3 = car_transform.forward().into();
    // Flipped 180° on the horizontal plane: camera now sits in front of the cart
    // looking back at it, instead of chasing from behind.
    let desired = car_transform.translation + forward * 3.5 + Vec3::Y * 1.6;

    let follow_speed = 6.0;
    camera_transform.translation = camera_transform
        .translation
        .lerp(desired, (follow_speed * time.delta_secs()).clamp(0.0, 1.0));

    let look_target = car_transform.translation + back * 2.0 + Vec3::Y * 0.6;
    camera_transform.look_at(look_target, Vec3::Y);
}

/// Backquote (`) OR clicking the on-screen DEBUG button flips the debug
/// coordinate-grid overlay on/off, showing or hiding every grid line and
/// number label in one go.
fn toggle_debug_mode(
    keys: Res<ButtonInput<KeyCode>>,
    mut debug_mode: ResMut<DebugMode>,
    mut entity_query: Query<&mut Visibility, With<DebugEntity>>,
    button_query: Query<&Interaction, (Changed<Interaction>, With<DebugToggleButton>)>,
    mut button_color_query: Query<&mut BackgroundColor, With<DebugToggleButton>>,
) {
    let key_pressed = keys.just_pressed(KeyCode::Backquote);
    let button_clicked = button_query
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed);

    if !key_pressed && !button_clicked {
        return;
    }

    debug_mode.0 = !debug_mode.0;
    let visibility = if debug_mode.0 {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut entity_query {
        *vis = visibility;
    }

    // Highlight the button itself while debug mode is active.
    if let Ok(mut color) = button_color_query.single_mut() {
        color.0 = if debug_mode.0 {
            Color::srgb(0.2, 0.7, 0.3)
        } else {
            Color::srgb(0.15, 0.15, 0.18)
        };
    }
}
