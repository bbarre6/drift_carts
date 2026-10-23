use avian3d::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::audio::Volume;
use bevy::camera::Viewport;
use bevy::prelude::*;
use bevy::render::mesh::PrimitiveTopology;
use bevy::window::PrimaryWindow;
use serde::{Deserialize, Serialize};
use std::f32::consts::{FRAC_PI_2, PI};
use std::fs;

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

/// How close the car needs to get to a checkpoint's (x, z) position to trigger
/// it, regardless of height -- generous enough to catch any line through the
/// corner, tight enough that the four checkpoints (see `checkpoint_positions`)
/// can't be hit out of order by accident.
const CHECKPOINT_RADIUS: f32 = TRACK_WIDTH * 1.5;
/// How many seconds the car must be getting steadily farther from its next
/// checkpoint, while actually moving, before the WRONG WAY banner shows.
const WRONG_WAY_TRIGGER_SECONDS: f32 = 1.2;

/// How many centerline points an AI car aims at ahead of its current nearest
/// point -- small enough to hug the road, large enough not to oversteer at
/// speed. See `ai_controller`.
const AI_LOOKAHEAD_POINTS: usize = 8;

/// Where the save file lives, relative to the working directory the game is
/// run from (see `load_save`/`save_game`).
const SAVE_PATH: &str = "save.json";

/// A sudden single-frame speed loss bigger than this (world units/s) is read
/// as "hit something solid" by `drift_score_system` -- there's no dependency
/// on avian3d's own collision events, just a before/after speed comparison on
/// the one car that can drift.
const WALL_HIT_DECEL_THRESHOLD: f32 = 10.0;

/// Top-level flow: a title card, the race itself, and a pause overlay reachable
/// from it. The world (track, car, HUD) is always spawned up front in `setup`;
/// what changes between states is just which input/UI systems are live and
/// which menu, if any, is on screen -- see the `OnEnter`/`OnExit` systems below.
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
enum GameState {
    #[default]
    Title,
    Playing,
    Paused,
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, PhysicsPlugins::default()))
        .init_state::<GameState>()
        .add_systems(Startup, (load_save, setup).chain())
        .init_resource::<DebugMode>()
        .init_resource::<Race>()
        .init_resource::<GhostRecording>()
        .init_resource::<DriftScore>()
        .init_resource::<Muted>()
        .add_systems(OnEnter(GameState::Title), spawn_title_menu)
        .add_systems(OnExit(GameState::Title), despawn_menu::<TitleMenuRoot>)
        .add_systems(OnEnter(GameState::Paused), spawn_pause_menu)
        .add_systems(OnExit(GameState::Paused), despawn_menu::<PauseMenuRoot>)
        .add_systems(
            Update,
            (
                title_buttons.run_if(in_state(GameState::Title)),
                pause_toggle.run_if(not(in_state(GameState::Title))),
                pause_buttons.run_if(in_state(GameState::Paused)),
            ),
        )
        .add_systems(
            Update,
            (
                car_controller,
                car_controller_p2,
                ai_controller,
                switch_car_model,
                toggle_multiplayer,
                ramp_zones,
                car_visual_tilt,
                drift_effects,
                drift_score_system,
                update_particles,
                race_checkpoints,
                record_ghost,
                update_ghost_visual,
                hazard_sweeper,
                engine_audio,
                tire_audio,
                update_hud,
                toggle_debug_mode,
                camera_follow,
            )
                .chain()
                .run_if(in_state(GameState::Playing)),
        )
        .add_systems(Update, save_on_exit)
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

/// Marks the car's physics body, so `switch_car_model` can find it to attach a
/// freshly spawned `CarVisual` child when the model changes.
#[derive(Component)]
struct CarBody;

/// Marks the visual-only glTF model child of the car's physics body. Its local
/// rotation gets a cosmetic pitch tilt (see `car_visual_tilt`) based on climb/fall
/// rate -- purely visual, so it can never feed back into the physics body itself.
#[derive(Component)]
struct CarVisual;

/// Which glTF model is currently shown for the car. Swappable at runtime (`C`
/// key, see `switch_car_model`) rather than a build-time choice, so both
/// models stay available side by side instead of one replacing the other.
#[derive(Resource, Clone, Copy, PartialEq, Eq)]
enum CarModel {
    F1,
    Kart,
    /// The Kenney kit's sedan -- a "cruiser" stat-swap rather than just a
    /// different look (see `stats`): lower grip and turn rate, but a
    /// meaningfully higher top speed. Unlocked by default; cycled in with
    /// the same `C` key as the other two.
    Sedan,
}

impl CarModel {
    fn next(self) -> Self {
        match self {
            CarModel::F1 => CarModel::Kart,
            CarModel::Kart => CarModel::Sedan,
            CarModel::Sedan => CarModel::F1,
        }
    }

    fn asset_path(self) -> &'static str {
        match self {
            CarModel::F1 => "models/f1_race_car.glb",
            CarModel::Kart => "models/race.glb",
            CarModel::Sedan => "models/sedan.glb",
        }
    }

    /// The `Car` handling stats that come with this model -- applied to the
    /// player's own car (see `switch_car_model`) whenever it's selected, so
    /// switching models is a real gameplay choice and not just a costume
    /// change. F1 and Kart intentionally keep the same tuned arcade feel
    /// they always had; only the Sedan trades grip for top speed.
    fn stats(self) -> Car {
        match self {
            CarModel::F1 | CarModel::Kart => Car::default(),
            CarModel::Sedan => Car {
                engine_power: 30.0,
                max_speed: 40.0,
                turn_speed: 2.2,
                drift_turn_multiplier: 1.3,
                grip: 7.0,
                drift_grip: 1.4,
            },
        }
    }

    /// Local transform placing the model's own wheel-contact point flush
    /// with the physics collider's bottom (body-local y = -0.3, see
    /// CAR_GROUND_CLEARANCE in `ramp_zones`), and correcting for whatever
    /// orientation/scale quirks each asset came with.
    fn visual_transform(self) -> Transform {
        match self {
            // f1_race_car.glb's root node carries a baked 90-degree rotation
            // about X (an artifact of the Blender OBJ->glTF conversion used
            // to produce this file) -- left alone, the car stands on its
            // nose instead of lying flat. Countering it with the exact
            // inverse here restores the orientation the mesh was actually
            // modeled in: lying flat, length along Z, nose toward -Z
            // (matching the physics body's own forward direction, same as
            // the kart). It's also modeled much bigger than the kart
            // (roughly 2.2x in every dimension) -- scaled down to stay
            // proportioned to the track/ramps/guardrails, which were all
            // sized around the kart. The glTF's own lowest point (its wheel
            // contact) sits at local y=0.16 rather than 0, so the Y offset
            // also folds in -0.16 * scale to bring the (scaled) wheels back
            // down flush with the physics collider's bottom.
            CarModel::F1 => Transform::from_xyz(0.0, -0.3 - 0.16 * 0.45, 0.0)
                .with_scale(Vec3::splat(0.45))
                .with_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
            // Same Kenney kit, same export conventions as the kart -- no
            // orientation correction needed, just its own clearance offset.
            CarModel::Kart | CarModel::Sedan => Transform::from_xyz(0.0, -0.3, 0.0),
        }
    }

    fn base_rotation(self) -> Quat {
        match self {
            CarModel::F1 => Quat::from_rotation_x(-FRAC_PI_2),
            CarModel::Kart | CarModel::Sedan => Quat::IDENTITY,
        }
    }
}

impl Default for CarModel {
    fn default() -> Self {
        CarModel::F1
    }
}

/// The model's own rest orientation, applied underneath the cosmetic tilt in
/// `car_visual_tilt`. Needed because that system *sets* (not composes with)
/// the visual's rotation every frame from the tilt angle alone -- without
/// this, any rotation baked into the model's own Transform at spawn (e.g. to
/// correct an asset's export orientation) gets slerped away within a
/// fraction of a second of the car existing.
#[derive(Component)]
struct CarVisualBaseRotation(Quat);

/// Spawns a `CarVisual` child of `parent` for `model`, using each model's own
/// tuned transform/base rotation (see `CarModel::visual_transform`). Shared
/// by the initial spawn in `setup` and by `switch_car_model` when the player
/// swaps models at runtime, so the two can never drift out of sync.
fn spawn_car_visual(
    commands: &mut Commands,
    parent: Entity,
    asset_server: &AssetServer,
    model: CarModel,
) {
    commands.entity(parent).with_children(|parent| {
        parent.spawn((
            CarVisual,
            WorldAssetRoot(
                asset_server.load(GltfAssetLabel::Scene(0).from_asset(model.asset_path())),
            ),
            model.visual_transform(),
            // See CarVisualBaseRotation's doc comment -- without this,
            // car_visual_tilt slerps the orientation correction above away
            // within a fraction of a second.
            CarVisualBaseRotation(model.base_rotation()),
        ));
    });
}

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

/// Marks the keyboard (WASD + Space) controlled car -- the one the camera,
/// HUD, drift particles/audio, and ghost-recording all follow. There is
/// always exactly one.
#[derive(Component)]
struct PlayerCar;

/// Marks the second local car, driven with the arrow keys + Right Ctrl,
/// spawned/despawned by `toggle_multiplayer` (the `M` key). Absent in
/// single-player.
#[derive(Component)]
struct Player2Car;

/// Marks an AI-controlled car and tracks its progress along `ai_path()`
/// (see `ai_controller`).
#[derive(Component)]
struct AiDriver {
    nearest_index: usize,
}

/// Tags a camera and which car it should follow -- `false` for the player 1
/// camera (always present), `true` for the player 2 camera (only present
/// while `Player2Car` is).
#[derive(Component)]
struct FollowCamera {
    follows_player2: bool,
}

/// Four points around the GROUND oval, in the order the player actually
/// drives them (see the car's start position/facing in `setup`): the
/// start/finish line, then each straight's far end in turn. A lap is defined
/// purely in terms of the ground oval -- the elevated level is a detour, not
/// part of the required loop.
fn checkpoint_positions() -> [Vec2; 4] {
    [
        Vec2::new(HALF_STRAIGHT, -TURN_RADIUS),
        Vec2::new(-HALF_STRAIGHT, -TURN_RADIUS),
        Vec2::new(-HALF_STRAIGHT, TURN_RADIUS),
        Vec2::new(HALF_STRAIGHT, TURN_RADIUS),
    ]
}

/// Race/lap state for the player: which checkpoint comes next, lap count,
/// current/best/last lap time, and the wrong-way timer. See
/// `race_checkpoints`.
#[derive(Resource)]
struct Race {
    next_checkpoint: usize,
    lap: u32,
    lap_start: f32,
    last_lap_time: Option<f32>,
    best_lap_time: Option<f32>,
    /// Distance to `next_checkpoint` as of the previous frame; `None` right
    /// after a checkpoint is hit, so the very next frame can't immediately
    /// read as "getting farther away".
    prev_checkpoint_distance: Option<f32>,
    wrong_way_timer: f32,
}

impl Default for Race {
    fn default() -> Self {
        Self {
            next_checkpoint: 1,
            lap: 0,
            lap_start: 0.0,
            last_lap_time: None,
            best_lap_time: None,
            prev_checkpoint_distance: None,
            wrong_way_timer: 0.0,
        }
    }
}

/// Recorded (time-into-lap, position, rotation) samples for the lap in
/// progress and for the best completed lap so far. `update_ghost_visual`
/// replays `best` in sync with the current lap's own clock; `record_ghost`
/// fills `current`, and `race_checkpoints` promotes it to `best` on a new
/// record (see that system).
#[derive(Resource, Default)]
struct GhostRecording {
    current: Vec<(f32, Vec3, Quat)>,
    best: Vec<(f32, Vec3, Quat)>,
}

/// Marks the translucent ghost car that replays `GhostRecording::best`.
#[derive(Component)]
struct GhostCar;

/// Charge-and-bank drift scoring: `current` fills while the player holds the
/// drift key and is actually sliding, banks into `total` on a clean release,
/// and is discarded (not banked) if `drift_score_system` detects a wall hit
/// first. See that system.
#[derive(Resource, Default)]
struct DriftScore {
    current: f32,
    total: f32,
}

/// Whether sound is currently muted. Carried in `PlaybackSettings::muted` for
/// every newly spawned sound, and applied to already-playing loops by
/// toggling every live `AudioSink` directly (see `pause_buttons`) -- changing
/// `GlobalVolume` alone does not affect sounds already playing.
#[derive(Resource, Default)]
struct Muted(bool);

/// Handles to every placeholder sound effect, loaded once in `setup`.
#[derive(Resource)]
struct AudioAssets {
    engine_loop: Handle<AudioSource>,
    tire_chirp: Handle<AudioSource>,
    drift_screech: Handle<AudioSource>,
    ramp_launch: Handle<AudioSource>,
    landing_thump: Handle<AudioSource>,
    lap_chime: Handle<AudioSource>,
    ambient_loop: Handle<AudioSource>,
}

/// Marks the player's own looping engine-sound entity, so `engine_audio` can
/// retarget its `AudioSink` speed/volume every frame without a fresh query.
#[derive(Component)]
struct EngineAudio;

/// Marks the looping drift-screech entity (only sounds while actually
/// drifting -- see `tire_audio`, which pauses/resumes its `AudioSink` rather
/// than despawning and respawning it every time the key toggles).
#[derive(Component)]
struct DriftAudio;

/// A slow-moving barrier on the elevated oval's far straight -- the "moving
/// hazard" feature: nothing to collide with here except the player's own
/// timing, since it has no physics collider (consistent with the ramps --
/// see `spawn_ramp`'s doc comment on why a collider next to a
/// kinematically-overridden car is a bad idea; this one just isn't on a car
/// at all, so it stays purely a dodge-it-visually hazard rather than a wall).
#[derive(Component)]
struct HazardSweeper {
    center_z: f32,
    amplitude: f32,
    speed: f32,
}

/// What gets written to/read from `SAVE_PATH`. Just the one best lap time for
/// now -- enough to make "best lap" mean something across separate play
/// sessions instead of resetting every launch.
#[derive(Serialize, Deserialize, Default)]
struct SaveData {
    best_lap_time: Option<f32>,
}

/// Reads `SAVE_PATH` (if it exists) before `setup` runs, seeding `Race` with
/// whatever best lap time was recorded last session. A missing or corrupt
/// save file is silently treated as "no save yet" -- there's nothing in it
/// worth failing startup over.
fn load_save(mut commands: Commands) {
    let data: SaveData = fs::read_to_string(SAVE_PATH)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    commands.insert_resource(Race {
        best_lap_time: data.best_lap_time,
        ..Race::default()
    });
}

/// Writes the current best lap time to `SAVE_PATH` once per frame it
/// changes. Keyed on the value itself (via `Local`) rather than a dirty
/// flag -- simpler, and saving a little too eagerly costs nothing for a file
/// this small.
fn save_on_exit(race: Res<Race>, mut last_saved: Local<Option<f32>>) {
    if race.best_lap_time == *last_saved {
        return;
    }
    *last_saved = race.best_lap_time;
    let data = SaveData {
        best_lap_time: race.best_lap_time,
    };
    if let Ok(text) = serde_json::to_string_pretty(&data) {
        let _ = fs::write(SAVE_PATH, text);
    }
}

/// Root node of whichever full-screen menu is currently up; both `TitleMenuRoot`
/// and `PauseMenuRoot` entities get despawned wholesale by `despawn_menu` on
/// `OnExit` of their state, so the two menus can never both linger at once.
#[derive(Component)]
struct TitleMenuRoot;

#[derive(Component)]
struct PauseMenuRoot;

#[derive(Component)]
struct StartButton;

#[derive(Component)]
struct ResumeButton;

#[derive(Component)]
struct RestartButton;

#[derive(Component)]
struct QuitButton;

#[derive(Component)]
struct MuteButton;

/// Marks every piece of the always-on-screen HUD text so `update_hud` can
/// find each line without re-deriving its layout.
#[derive(Component)]
enum HudLine {
    Speed,
    Lap,
    LapTime,
    BestLap,
    WrongWay,
    DriftScore,
}

/// Despawns whatever root entity of type `T` exists (recursively, so its
/// children go with it). Shared by both menus' `OnExit` system -- see the
/// doc comment on `TitleMenuRoot`.
fn despawn_menu<T: Component>(mut commands: Commands, query: Query<Entity, With<T>>) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}

/// A plain button styled consistently for both the title and pause menus.
fn menu_button(parent: &mut ChildSpawnerCommands<'_>, label: &str, marker: impl Component) {
    parent
        .spawn((
            marker,
            Button,
            Node {
                width: Val::Px(220.0),
                height: Val::Px(48.0),
                margin: UiRect::all(Val::Px(8.0)),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.18, 0.18, 0.22)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(label),
                TextFont {
                    font_size: 20.0.into(),
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
        });
}

/// Spawned on entering `GameState::Title`: the game's name plus a Start
/// button. Despawned wholesale on leaving the state (see `despawn_menu`).
fn spawn_title_menu(mut commands: Commands) {
    commands
        .spawn((
            TitleMenuRoot,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                right: Val::Px(0.0),
                bottom: Val::Px(0.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                row_gap: Val::Px(18.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.05, 0.08, 0.92)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("DRIFT CARTS"),
                TextFont {
                    font_size: 56.0.into(),
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
            parent.spawn((
                Text::new("WASD to drive, Space to drift, M for 2-player"),
                TextFont {
                    font_size: 16.0.into(),
                    ..default()
                },
                TextColor(Color::srgb(0.75, 0.75, 0.8)),
            ));
            menu_button(parent, "START", StartButton);
        });
}

/// Spawned on entering `GameState::Paused`: Resume / Restart / Quit, plus a
/// mute toggle.
fn spawn_pause_menu(mut commands: Commands, muted: Res<Muted>) {
    commands
        .spawn((
            PauseMenuRoot,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                right: Val::Px(0.0),
                bottom: Val::Px(0.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                row_gap: Val::Px(14.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.05, 0.08, 0.85)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("PAUSED"),
                TextFont {
                    font_size: 40.0.into(),
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
            menu_button(parent, "RESUME", ResumeButton);
            menu_button(parent, "RESTART", RestartButton);
            menu_button(
                parent,
                if muted.0 { "UNMUTE" } else { "MUTE" },
                MuteButton,
            );
            menu_button(parent, "QUIT", QuitButton);
        });
}

/// Click handling for the title screen's one button.
fn title_buttons(
    mut next_state: ResMut<NextState<GameState>>,
    query: Query<&Interaction, (Changed<Interaction>, With<StartButton>)>,
) {
    if query
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        next_state.set(GameState::Playing);
    }
}

/// Escape toggles Playing <-> Paused (never fires from the title screen,
/// since the whole Update set this lives outside of `run_if`s on
/// `not(in_state(Title))`).
fn pause_toggle(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<GameState>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if !keys.just_pressed(KeyCode::Escape) {
        return;
    }
    match state.get() {
        GameState::Playing => next_state.set(GameState::Paused),
        GameState::Paused => next_state.set(GameState::Playing),
        GameState::Title => {}
    }
}

/// Click handling for the pause menu's four buttons.
fn pause_buttons(
    mut next_state: ResMut<NextState<GameState>>,
    mut app_exit: MessageWriter<AppExit>,
    mut muted: ResMut<Muted>,
    mut sink_query: Query<&mut AudioSink>,
    resume_query: Query<&Interaction, (Changed<Interaction>, With<ResumeButton>)>,
    restart_query: Query<&Interaction, (Changed<Interaction>, With<RestartButton>)>,
    quit_query: Query<&Interaction, (Changed<Interaction>, With<QuitButton>)>,
    mute_query: Query<&Interaction, (Changed<Interaction>, With<MuteButton>)>,
    mut player_query: Query<
        (&mut Transform, &mut LinearVelocity, &mut AngularVelocity),
        With<PlayerCar>,
    >,
    mut race: ResMut<Race>,
    mut ghost: ResMut<GhostRecording>,
    time: Res<Time>,
) {
    if resume_query.iter().any(|i| *i == Interaction::Pressed) {
        next_state.set(GameState::Playing);
    }

    if restart_query.iter().any(|i| *i == Interaction::Pressed) {
        if let Ok((mut transform, mut lin_vel, mut ang_vel)) = player_query.single_mut() {
            transform.translation = Vec3::new(HALF_STRAIGHT, 1.0, -TURN_RADIUS);
            transform.rotation = Transform::from_translation(transform.translation)
                .looking_to(Vec3::NEG_X, Vec3::Y)
                .rotation;
            lin_vel.0 = Vec3::ZERO;
            ang_vel.0 = Vec3::ZERO;
        }
        *race = Race {
            best_lap_time: race.best_lap_time,
            lap_start: time.elapsed_secs(),
            ..Race::default()
        };
        ghost.current.clear();
        next_state.set(GameState::Playing);
    }

    if quit_query.iter().any(|i| *i == Interaction::Pressed) {
        app_exit.write(AppExit::Success);
    }

    if mute_query.iter().any(|i| *i == Interaction::Pressed) {
        muted.0 = !muted.0;
        for mut sink in &mut sink_query {
            sink.toggle_mute();
        }
        // `set` (not `set_if_neq`) re-runs OnExit/OnEnter even though the
        // state value itself isn't changing -- the simplest way to get the
        // pause menu's MUTE/UNMUTE label to redraw immediately.
        next_state.set(GameState::Paused);
    }
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

    // Elevated oval's physical floor: a solid platform under its whole
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
    let floor_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.5, 0.52),
        perceptual_roughness: 0.9,
        ..default()
    });

    // Flush with where the connector ramps hand off (see CONNECTOR_RAMP_LENGTH's
    // doc comment): the ramps already provide the climbing surface for x below
    // this, at their own z-bands, so the floor only needs to start here for
    // those bands -- starting it back at the near turn's full apex reach
    // (elevated_floor_half_x) there instead hung a flat slab in midair directly
    // over the still-rising ramp surface, a horizontal overhang the debug grid's
    // numbered lines made easy to spot sitting between world-x 70 and 80.
    let near_straight_x = ELEVATED_OFFSET_X - HALF_STRAIGHT;
    let near_edge_x = ELEVATED_OFFSET_X - elevated_floor_half_x;
    let far_edge_x = ELEVATED_OFFSET_X + elevated_floor_half_x;
    // Width of the non-ramp strip down the middle (the turn's apex/infield),
    // narrowed in from the full Z half-extent by each ramp's own half-width so
    // this piece stops exactly at the ramp bands instead of overlapping them.
    let apex_half_z = TURN_RADIUS - TRACK_WIDTH / 2.0;

    // Near turn's apex/infield -- the only part of the near end that isn't
    // directly above a connector ramp.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(
            near_straight_x - near_edge_x,
            0.08,
            apex_half_z * 2.0,
        ))),
        MeshMaterial3d(floor_material.clone()),
        Transform::from_xyz((near_edge_x + near_straight_x) / 2.0, ELEVATED_HEIGHT, 0.0),
    ));
    commands.spawn((
        RigidBody::Static,
        Collider::cuboid(near_straight_x - near_edge_x, 0.2, apex_half_z * 2.0),
        Transform::from_xyz(
            (near_edge_x + near_straight_x) / 2.0,
            ELEVATED_HEIGHT - 0.1,
            0.0,
        ),
    ));

    // Rest of the platform: both straights (full width, ramps safely behind
    // this edge by now), the far turn, and the far-side infield.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(
            far_edge_x - near_straight_x,
            0.08,
            elevated_floor_half_z * 2.0,
        ))),
        MeshMaterial3d(floor_material),
        Transform::from_xyz((near_straight_x + far_edge_x) / 2.0, ELEVATED_HEIGHT, 0.0),
    ));
    commands.spawn((
        RigidBody::Static,
        Collider::cuboid(
            far_edge_x - near_straight_x,
            0.2,
            elevated_floor_half_z * 2.0,
        ),
        Transform::from_xyz(
            (near_straight_x + far_edge_x) / 2.0,
            ELEVATED_HEIGHT - 0.1,
            0.0,
        ),
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

    // Car physics body (invisible box) with the active CarModel as a visual-only
    // child (see spawn_car_visual) -- swappable at runtime with the `C` key
    // (switch_car_model). Starts at the opposite (east) end of the bottom
    // straight, facing -X, so the loop is driven the other way around. Drops
    // in from y=1 so it settles onto the road on its own.
    let start_pos = Vec3::new(HALF_STRAIGHT, 1.0, -TURN_RADIUS);
    let car_model = CarModel::default();
    let car_body = commands
        .spawn((
            CarBody,
            PlayerCar,
            RampState::default(),
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
    commands.insert_resource(car_model);
    spawn_car_visual(&mut commands, car_body, &asset_server, car_model);

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Chase camera; camera_follow() drives it every frame. Full-window by
    // default; toggle_multiplayer resizes it to the top half if/when a
    // second player joins.
    commands.spawn((
        FollowCamera {
            follows_player2: false,
        },
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

    // Checkpoint markers: a thin disc flat on the road at each of the four
    // points `checkpoint_positions` defines, so the "next expected"
    // checkpoint (tracked by `Race`, highlighted by `race_checkpoints`) is
    // visible rather than an invisible gate. White + translucent at rest.
    let checkpoint_mesh = meshes.add(Cylinder::new(TRACK_WIDTH * 0.45, 0.06));
    for (i, point) in checkpoint_positions().into_iter().enumerate() {
        commands.spawn((
            CheckpointVisual(i),
            Mesh3d(checkpoint_mesh.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgba(0.9, 0.9, 0.9, 0.35),
                alpha_mode: AlphaMode::Blend,
                emissive: LinearRgba::rgb(0.3, 0.3, 0.3),
                unlit: true,
                ..default()
            })),
            Transform::from_xyz(point.x, 0.15, point.y),
        ));
    }

    // Translucent ghost car: replays the best lap recorded so far (see
    // `GhostRecording`/`update_ghost_visual`), hidden until one exists.
    let ghost_mesh = meshes.add(Cuboid::new(1.0, 0.6, 2.0));
    commands.spawn((
        GhostCar,
        Visibility::Hidden,
        Mesh3d(ghost_mesh),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.3, 0.7, 1.0, 0.35),
            alpha_mode: AlphaMode::Blend,
            emissive: LinearRgba::rgb(0.1, 0.25, 0.4),
            unlit: true,
            ..default()
        })),
        Transform::from_translation(start_pos),
    ));

    // Rammable AI traffic: dumb, non-racing obstacles -- plain dynamic
    // bodies with no driver at all, scattered down both straights so
    // there's something to knock around mid-drift.
    let cone_mesh = meshes.add(Cone {
        radius: 0.35,
        height: 0.7,
    });
    let cone_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.45, 0.05),
        ..default()
    });
    for straight_z in [-TURN_RADIUS, TURN_RADIUS] {
        for t in [-0.6_f32, -0.2, 0.2, 0.6] {
            let x = t * HALF_STRAIGHT;
            // Offset off the centerline so cones don't sit exactly on the
            // racing line every time -- still well within TRACK_WIDTH.
            let z_offset = if t > 0.0 { 2.0 } else { -2.0 };
            commands.spawn((
                TrafficProp,
                RigidBody::Dynamic,
                Collider::cylinder(0.35, 0.7),
                Mass(2.0),
                Mesh3d(cone_mesh.clone()),
                MeshMaterial3d(cone_material.clone()),
                Transform::from_xyz(x, 0.35, straight_z + z_offset),
            ));
        }
    }

    // Moving hazard: a low barrier sweeping back and forth across the
    // elevated oval's far straight (see `hazard_sweeper`). Visual-only, like
    // the ramps -- nothing here to physically collide with, just something
    // worth timing a drift around.
    commands.spawn((
        HazardSweeper {
            center_z: ELEVATED_HEIGHT * 0.0 + TURN_RADIUS,
            amplitude: TRACK_WIDTH * 0.35,
            speed: 0.6,
        },
        Mesh3d(meshes.add(Cuboid::new(0.4, 1.0, TRACK_WIDTH * 0.5))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.75, 0.1),
            emissive: LinearRgba::rgb(0.5, 0.4, 0.0),
            ..default()
        })),
        Transform::from_xyz(
            ELEVATED_OFFSET_X,
            ELEVATED_HEIGHT + 0.5,
            TURN_RADIUS,
        ),
    ));

    // Infield shortcut: a small jump ramp cutting across the ground oval's
    // middle, from just past the start/finish line straight across to the
    // far straight -- a real time save, but only if you're confident enough
    // to leave the road for it.
    spawn_ramp(
        &mut commands,
        &mut meshes,
        &mut materials,
        HALF_STRAIGHT * 0.3,
        -TURN_RADIUS,
        -1.0,
        4.0,
        1.2,
        4.0,
        Color::srgb(0.7, 0.55, 0.15),
        true,
    );

    // AI racers: a couple of bots following the ground oval's own road
    // centerline (see `ai_path`/`ai_controller`), each with slightly
    // different stats for a bit of personality.
    let ai_path = ai_path();
    let ai_starts = [
        (ai_path[5], Car {
            engine_power: 30.0,
            max_speed: 29.0,
            ..Car::default()
        }),
        (ai_path[ai_path.len() / 2 + 5], Car {
            engine_power: 37.0,
            max_speed: 35.0,
            grip: 11.0,
            ..Car::default()
        }),
    ];
    for (start, stats) in ai_starts {
        let facing = {
            let next = ai_path[(ai_path.iter().position(|p| *p == start).unwrap_or(0) + 1)
                % ai_path.len()];
            (next - start).normalize_or(Vec3::NEG_X)
        };
        let ai_body = commands
            .spawn((
                AiDriver { nearest_index: 0 },
                RampState::default(),
                stats,
                RigidBody::Dynamic,
                Collider::cuboid(1.0, 0.6, 2.0),
                LinearVelocity::ZERO,
                AngularVelocity::ZERO,
                LockedAxes::new().lock_rotation_x().lock_rotation_z(),
                MaxAngularSpeed(6.0),
                Transform::from_translation(start + Vec3::Y * 0.5).looking_to(facing, Vec3::Y),
                Visibility::default(),
            ))
            .id();
        spawn_car_visual(&mut commands, ai_body, &asset_server, CarModel::Kart);
    }

    // Audio: load every placeholder sound once, and start the two ambient
    // loops (background music + the player's own engine note) right away.
    // See engine_audio/tire_audio for how the engine and drift-screech loops
    // get retargeted every frame, and generate_audio.py (run once, offline)
    // for how the placeholder clips themselves were made.
    let audio = AudioAssets {
        engine_loop: asset_server.load("audio/engine_loop.wav"),
        tire_chirp: asset_server.load("audio/tire_chirp.wav"),
        drift_screech: asset_server.load("audio/drift_screech.wav"),
        ramp_launch: asset_server.load("audio/ramp_launch.wav"),
        landing_thump: asset_server.load("audio/landing_thump.wav"),
        lap_chime: asset_server.load("audio/lap_chime.wav"),
        ambient_loop: asset_server.load("audio/ambient_loop.wav"),
    };
    commands.spawn((
        EngineAudio,
        AudioPlayer(audio.engine_loop.clone()),
        PlaybackSettings::LOOP.with_volume(Volume::Linear(0.5)),
    ));
    commands.spawn((
        DriftAudio,
        AudioPlayer(audio.drift_screech.clone()),
        PlaybackSettings::LOOP
            .with_volume(Volume::Linear(0.0))
            .paused(),
    ));
    commands.spawn((
        AudioPlayer(audio.ambient_loop.clone()),
        PlaybackSettings::LOOP.with_volume(Volume::Linear(0.25)),
    ));
    commands.insert_resource(audio);

    // HUD: speed, lap count, current/best lap time, drift score, and a
    // WRONG WAY banner that's invisible (zero alpha) until race_checkpoints
    // actually needs it.
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            right: Val::Px(10.0),
            top: Val::Px(10.0),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::FlexEnd,
            row_gap: Val::Px(4.0),
            ..default()
        })
        .with_children(|parent| {
            let line = |parent: &mut ChildSpawnerCommands<'_>, marker: HudLine, text: &str| {
                parent.spawn((
                    marker,
                    Text::new(text),
                    TextFont {
                        font_size: 18.0.into(),
                        ..default()
                    },
                    TextColor(Color::WHITE),
                ));
            };
            line(parent, HudLine::Speed, "SPEED 0");
            line(parent, HudLine::Lap, "LAP 0");
            line(parent, HudLine::LapTime, "TIME 0.0");
            line(parent, HudLine::BestLap, "BEST --");
            line(parent, HudLine::DriftScore, "DRIFT 0");
        });
    commands.spawn((
        HudLine::WrongWay,
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            right: Val::Px(0.0),
            top: Val::Px(60.0),
            justify_content: JustifyContent::Center,
            ..default()
        },
        Text::new(""),
        TextFont {
            font_size: 28.0.into(),
            ..default()
        },
        TextColor(Color::srgba(1.0, 0.2, 0.2, 0.0)),
    ));
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
}

/// Which ramp (if any) a car is currently inside, tracked per CAR rather
/// than per ramp (see `RampZone`'s old `car_was_inside`, now gone) so
/// multiple cars can use ramps independently without stomping each other's
/// enter/exit transitions. Attached to every driveable car.
#[derive(Component, Default)]
struct RampState {
    inside: Option<Entity>,
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

    // Outward-wound triangles: bottom, sloped top, back wall, plus two side
    // walls -- but only for short jump ramps, where they're an unnoticeable
    // small triangle. At a connector ramp's full length (70 units) those same
    // side walls rendered as a huge flat diagonal slab looming up alongside
    // the ramp on both sides, a flappy-looking overhang with nothing to do
    // with the actual drivable surface. Connector ramps already get their own
    // guardrails (see below), so they're dropped entirely there.
    let mut triangles = vec![
        [front_left, back_left, back_right],
        [front_left, back_right, front_right],
        [front_left, front_right, top_right],
        [front_left, top_right, top_left],
        [back_left, top_right, back_right],
        [back_left, top_left, top_right],
    ];
    if is_jump {
        triangles.push([front_left, top_left, back_left]);
        triangles.push([front_right, back_right, top_right]);
    }
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
        },
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(ramp_material),
        Transform::from_translation(origin).with_rotation(rotation),
    ));

    // Connector ramps (not jump ramps) get guardrails along both sides,
    // visual-only like the ramp surface itself (see this function's own doc
    // comment above: a real collider right next to a car whose Y is
    // kinematically overridden every frame fights the physics solver --
    // that's exactly why the ramp has no collider of its own either). The
    // "can't drift off the side" guarantee is instead enforced kinematically
    // in `ramp_zones`, by clamping the car's sideways position to the ramp's
    // width -- see the comment there.
    //
    // These used to be plain vertical walls spanning the full climb height
    // (12 units tall, to guarantee overlapping the car's height everywhere
    // along the climb without needing to track the slope) -- simple, but at a
    // connector ramp's full 70-unit length that was a towering flat slab
    // right next to the camera the whole way up, a looming "flap" alongside
    // the ramp with nothing to do with the actual rail. Tilting a low rail to
    // follow the slope instead hugs the ramp surface at a normal barrier-like
    // height while still reading as a continuous rail along the climb.
    if !is_jump {
        const GUARDRAIL_HEIGHT: f32 = 1.2;
        let slope_len = length.hypot(height);
        // Angle of the climb; tilts the rail to lie flush along the ramp's
        // sloped top surface instead of standing bolt upright.
        let tilt_angle = height.atan2(length);
        let tilt = Quat::from_rotation_z(tilt_angle);
        // Unit vector along the incline and its outward (up-and-back) normal,
        // both in the ramp's local X-Y plane, used to lift the rail's center
        // up out of the ramp surface by half its own height.
        let incline_dir = Vec2::new(length, height) / slope_len;
        let normal = Vec2::new(-incline_dir.y, incline_dir.x);
        let mid = Vec2::new(length / 2.0, height / 2.0) + normal * (GUARDRAIL_HEIGHT / 2.0);
        let guardrail_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.22, 0.22, 0.25),
            perceptual_roughness: 0.8,
            ..default()
        });
        for side in [-1.0_f32, 1.0] {
            let local_pos = Vec3::new(mid.x, mid.y, side * (half_width + 0.15));
            let world_pos = origin + rotation * local_pos;
            let world_rotation = rotation * tilt;
            commands.spawn((
                Mesh3d(meshes.add(Cuboid::new(slope_len, GUARDRAIL_HEIGHT, 0.3))),
                MeshMaterial3d(guardrail_material.clone()),
                Transform::from_translation(world_pos).with_rotation(world_rotation),
            ));
        }
    }
}

/// Lifts a car's height to match a ramp's slope while it's within the ramp's
/// footprint -- no collision, just directly following the known profile --
/// and gives it a clean upward velocity boost the moment it crosses the far
/// edge. Runs over every driveable car (player, player 2, AI), each tracking
/// its own ramp occupancy via `RampState` so they can't interfere with one
/// another's enter/exit transitions.
///
/// See the comment on `spawn_ramp` for why this isn't done via a real collider.
/// `PlaybackSettings::muted()` is a parameterless "start muted" builder --
/// this is the conditional version, used for every one-shot sound so a clip
/// spawned while the game is already muted doesn't get a moment of sound
/// before anything would otherwise toggle it.
fn one_shot(mut settings: PlaybackSettings, muted: bool) -> PlaybackSettings {
    settings.muted = muted;
    settings
}

fn ramp_zones(
    mut commands: Commands,
    ramp_query: Query<(Entity, &RampZone)>,
    mut car_query: Query<(&mut Transform, &mut LinearVelocity, &mut RampState)>,
    audio: Option<Res<AudioAssets>>,
    muted: Res<Muted>,
) {
    // Half the car collider's height (see Collider::cuboid(1.0, 0.6, 2.0) in
    // setup) -- how far above the ramp's surface the car's center should sit,
    // matching how it normally rests on flat ground.
    const CAR_GROUND_CLEARANCE: f32 = 0.3;

    // How far past a connector ramp's own width the kinematic "guardrail"
    // below still catches the car -- wide enough to recover a car drifting
    // off the edge mid-climb, narrow enough that a car elsewhere on the
    // track (e.g. the infield, well off to the side) within the same X range
    // is never mistaken for being on the ramp.
    const DRIFT_MARGIN: f32 = 2.0;
    const LAUNCH_BOOST_SCALE: f32 = 0.35;
    const MAX_LAUNCH_SPEED: f32 = 6.0;

    for (mut transform, mut lin_vel, mut ramp_state) in &mut car_query {
        let mut now_inside: Option<Entity> = None;

        for (ramp_entity, ramp) in &ramp_query {
            let local = ramp.inverse_rotation * (transform.translation - ramp.origin);
            let within_length = local.x >= 0.0 && local.x <= ramp.length;
            let within_width = local.z.abs() <= ramp.half_width;
            let within_margin = local.z.abs() <= ramp.half_width + DRIFT_MARGIN;
            let inside = within_length && (within_width || (!ramp.is_jump && within_margin));
            if !inside {
                continue;
            }
            now_inside = Some(ramp_entity);

            let climb = (local.x / ramp.length).clamp(0.0, 1.0);
            transform.translation.y = ramp.origin.y + climb * ramp.height + CAR_GROUND_CLEARANCE;
            // Riding the known slope, not falling -- let gravity resume once it exits.
            lin_vel.0.y = 0.0;

            // Connector ramps act as their own guardrail: once drifting past
            // the rail, clamp straight back to the edge instead of relying on
            // a real collider there (see spawn_ramp's doc comment -- a static
            // collider right next to a car whose Y is overridden every frame
            // like this fought the physics solver and left the car stuck).
            if !within_width {
                let clamped_z = local.z.clamp(-ramp.half_width, ramp.half_width);
                let rotation = ramp.inverse_rotation.inverse();
                let corrected =
                    ramp.origin + rotation * Vec3::new(local.x, local.y, clamped_z);
                transform.translation.x = corrected.x;
                transform.translation.z = corrected.z;
                // Cancel sideways velocity too, so the car doesn't keep
                // pushing into the rail every frame.
                let world_z_axis: Vec3 = rotation * Vec3::Z;
                let lateral_speed = lin_vel.0.dot(world_z_axis);
                lin_vel.0 -= world_z_axis * lateral_speed;
            }

            if ramp_state.inside.is_none() && ramp.is_jump {
                // Just entered a jump ramp's footprint: anticipation whoosh.
                if let Some(audio) = audio.as_ref() {
                    commands.spawn((
                        AudioPlayer(audio.ramp_launch.clone()),
                        one_shot(
                            PlaybackSettings::DESPAWN.with_volume(Volume::Linear(0.5)),
                            muted.0,
                        ),
                    ));
                }
            }
            break; // a car is never meaningfully inside two ramps at once
        }

        if let (None, Some(prev_ramp)) = (now_inside, ramp_state.inside) {
            // Just crossed the far edge of whatever ramp we were just on.
            if let Ok((_, ramp)) = ramp_query.get(prev_ramp) {
                if ramp.is_jump {
                    // JUMP ramp: launch, trading none of the horizontal speed that
                    // was preserved while climbing -- it was the ongoing loss of
                    // that speed to the ramp's own collision drag that this whole
                    // approach exists to avoid.
                    //
                    // The raw version of this (speed * sin(ramp angle), no scaling)
                    // jumped much higher once top speed was raised, since it
                    // scales directly with how fast the car is going.
                    // LAUNCH_BOOST_SCALE tones that down to a smaller hop
                    // regardless of how fast top speed gets tuned later, and
                    // MAX_LAUNCH_SPEED caps it outright so it can't scale away
                    // again.
                    let horizontal_speed = Vec2::new(lin_vel.0.x, lin_vel.0.z).length();
                    let launch_angle = ramp.height.atan2(ramp.length);
                    let boost = horizontal_speed * launch_angle.sin() * LAUNCH_BOOST_SCALE;
                    lin_vel.0.y += boost.min(MAX_LAUNCH_SPEED);

                    if let Some(audio) = audio.as_ref() {
                        commands.spawn((
                            AudioPlayer(audio.landing_thump.clone()),
                            one_shot(
                                PlaybackSettings::DESPAWN.with_volume(Volume::Linear(0.6)),
                                muted.0,
                            ),
                        ));
                    }
                }
                // A connector ramp (is_jump == false) skips all of this: it already
                // ends exactly at the new level's resting height (see
                // CAR_GROUND_CLEARANCE above), so the car just rolls off onto flat
                // ground with no launch and no thump.
            }
        }

        ramp_state.inside = now_inside;
    }
}

/// Core arcade handling math, shared by both local players (see
/// `car_controller` and `car_controller_p2`, which just differ in which keys
/// they read) and reused as-is regardless of which car it's driving.
fn drive_car(
    car: &Car,
    transform: &Transform,
    lin_vel: &mut LinearVelocity,
    ang_vel: &mut AngularVelocity,
    throttle: f32,
    steer: f32,
    drifting: bool,
    dt: f32,
) {
    let forward: Vec3 = transform.forward().into();
    let right: Vec3 = transform.right().into();

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

/// Reads throttle/steer/drift out of whichever three-state key groups a
/// player is using (see `car_controller`/`car_controller_p2`).
fn read_input(
    keys: &ButtonInput<KeyCode>,
    up: KeyCode,
    down: KeyCode,
    left: KeyCode,
    right: KeyCode,
    drift: KeyCode,
) -> (f32, f32, bool) {
    // Inverted: "up" reverses, "down" accelerates forward -- matches the
    // original single-player binding this was split out of.
    let throttle = if keys.pressed(up) {
        -1.0
    } else if keys.pressed(down) {
        1.0
    } else {
        0.0
    };
    let steer = if keys.pressed(left) {
        -1.0
    } else if keys.pressed(right) {
        1.0
    } else {
        0.0
    };
    (throttle, steer, keys.pressed(drift))
}

/// Player 1: WASD to drive, Space to drift. Always present.
fn car_controller(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<
        (&Car, &Transform, &mut LinearVelocity, &mut AngularVelocity),
        With<PlayerCar>,
    >,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let (throttle, steer, drifting) = read_input(
        &keys,
        KeyCode::KeyW,
        KeyCode::KeyS,
        KeyCode::KeyA,
        KeyCode::KeyD,
        KeyCode::Space,
    );
    for (car, transform, mut lin_vel, mut ang_vel) in &mut query {
        drive_car(
            car, transform, &mut lin_vel, &mut ang_vel, throttle, steer, drifting, dt,
        );
    }
}

/// Player 2 (only exists once `toggle_multiplayer` has spawned one): arrow
/// keys to drive, Right Ctrl to drift -- kept fully separate from player 1's
/// keys so both can drive at once in split-screen.
fn car_controller_p2(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<
        (&Car, &Transform, &mut LinearVelocity, &mut AngularVelocity),
        With<Player2Car>,
    >,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let (throttle, steer, drifting) = read_input(
        &keys,
        KeyCode::ArrowUp,
        KeyCode::ArrowDown,
        KeyCode::ArrowLeft,
        KeyCode::ArrowRight,
        KeyCode::ControlRight,
    );
    for (car, transform, mut lin_vel, mut ang_vel) in &mut query {
        drive_car(
            car, transform, &mut lin_vel, &mut ang_vel, throttle, steer, drifting, dt,
        );
    }
}

/// The ground oval's road centerline, in the direction the player actually
/// drives it (see the car's start position/facing in `setup`) -- `
/// track_centerline` itself lists points the other way around the loop, so
/// this just reverses it. Shared by AI steering (`ai_controller`) and the
/// AI start positions chosen in `setup`.
fn ai_path() -> Vec<Vec3> {
    let mut points = track_centerline(HALF_STRAIGHT, TURN_RADIUS);
    points.reverse();
    points
}

/// Simple waypoint-following AI: each bot finds the nearest point on
/// `ai_path()`, steers toward a point a few indices ahead of it (so it cuts
/// in on corners rather than tracking exactly behind itself), and always
/// throttles forward -- no drifting, no avoidance, just a rolling obstacle
/// that roughly races the line.
fn ai_controller(
    time: Res<Time>,
    mut query: Query<(&Car, &Transform, &mut LinearVelocity, &mut AngularVelocity, &mut AiDriver)>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let path = ai_path();

    for (car, transform, mut lin_vel, mut ang_vel, mut driver) in &mut query {
        let pos = transform.translation;
        // Search near the last known index first (cars don't teleport), so
        // this stays cheap even though it's technically O(n) worst case.
        let (nearest, _) = path
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.distance_squared(pos)
                    .partial_cmp(&b.distance_squared(pos))
                    .unwrap()
            })
            .unwrap();
        driver.nearest_index = nearest;
        let target = path[(nearest + AI_LOOKAHEAD_POINTS) % path.len()];

        let to_target = (target - pos).with_y(0.0);
        let forward: Vec3 = transform.forward().into();
        let flat_forward = forward.with_y(0.0).normalize_or(Vec3::NEG_X);
        let heading_error = flat_forward
            .angle_between(to_target.normalize_or(flat_forward))
            * flat_forward.cross(to_target).y.signum();

        let steer = (heading_error * 2.0).clamp(-1.0, 1.0);
        // Ease off the throttle in sharper corners -- a flat-out AI cuts
        // every apex wide and looks less like it's actually racing.
        let throttle = 1.0 - (heading_error.abs() / FRAC_PI_2).clamp(0.0, 0.7);

        drive_car(
            car, transform, &mut lin_vel, &mut ang_vel, throttle, steer, false, dt,
        );
    }
}

/// `M` spawns or despawns the second local player and resizes both cameras
/// to a top/bottom split (or back to one full-window camera). Viewport
/// sizes are computed once at toggle time from the window's current size --
/// they don't follow a later resize, which is a fine trade for how rarely
/// this actually gets toggled mid-session.
fn toggle_multiplayer(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    asset_server: Res<AssetServer>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    player2_query: Query<Entity, With<Player2Car>>,
    mut camera_query: Query<(&FollowCamera, &mut Camera)>,
) {
    if !keys.just_pressed(KeyCode::KeyM) {
        return;
    }
    let Ok(window) = window_query.single() else {
        return;
    };
    let size = window.physical_size();

    if let Ok(existing) = player2_query.single() {
        commands.entity(existing).despawn();
        for (follow, mut camera) in &mut camera_query {
            if follow.follows_player2 {
                camera.is_active = false;
            } else {
                camera.viewport = None;
            }
        }
        return;
    }

    let half = UVec2::new(size.x, size.y / 2);
    for (follow, mut camera) in &mut camera_query {
        camera.is_active = true;
        camera.viewport = Some(Viewport {
            physical_position: UVec2::new(0, if follow.follows_player2 { half.y } else { 0 }),
            physical_size: half,
            depth: 0.0..1.0,
        });
    }

    let start_pos = Vec3::new(HALF_STRAIGHT, 1.0, TURN_RADIUS);
    let body = commands
        .spawn((
            Player2Car,
            RampState::default(),
            Car::default(),
            RigidBody::Dynamic,
            Collider::cuboid(1.0, 0.6, 2.0),
            LinearVelocity::ZERO,
            AngularVelocity::ZERO,
            LockedAxes::new().lock_rotation_x().lock_rotation_z(),
            MaxAngularSpeed(6.0),
            Transform::from_translation(start_pos).looking_to(Vec3::X, Vec3::Y),
            Visibility::default(),
        ))
        .id();
    spawn_car_visual(&mut commands, body, &asset_server, CarModel::Kart);

    commands.spawn((
        FollowCamera {
            follows_player2: true,
        },
        Camera3d::default(),
        Camera {
            order: 1,
            viewport: Some(Viewport {
                physical_position: UVec2::new(0, half.y),
                physical_size: half,
                depth: 0.0..1.0,
            }),
            ..default()
        },
        Transform::from_xyz(0.0, 2.2, -4.0).looking_at(Vec3::new(0.0, 0.8, 0.0), Vec3::Y),
    ));
}

/// Marks each of the four checkpoint discs spawned in `setup`, indexed to
/// match `checkpoint_positions`/`Race::next_checkpoint`.
#[derive(Component)]
struct CheckpointVisual(usize);

/// Marks a dumb, non-driving rammable obstacle (see `setup`).
#[derive(Component)]
struct TrafficProp;

/// Lap/checkpoint tracking, wrong-way detection, and checkpoint-disc
/// highlighting, all in one pass over the player's own position each frame.
/// See `Race`'s doc comment for the state machine, and `checkpoint_positions`
/// for why checkpoint 0 doubles as the start/finish line.
fn race_checkpoints(
    time: Res<Time>,
    mut race: ResMut<Race>,
    mut ghost: ResMut<GhostRecording>,
    player_query: Query<&Transform, With<PlayerCar>>,
    mut checkpoint_query: Query<(&CheckpointVisual, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut wrong_way_query: Query<&mut TextColor, With<HudLine>>,
    audio: Option<Res<AudioAssets>>,
    mut commands: Commands,
    muted: Res<Muted>,
) {
    let Ok(transform) = player_query.single() else {
        return;
    };
    let pos = Vec2::new(transform.translation.x, transform.translation.z);
    let checkpoints = checkpoint_positions();
    let target = checkpoints[race.next_checkpoint];
    let dist = pos.distance(target);

    // Wrong-way: the distance to our own next checkpoint has been growing,
    // steadily, while the car is actually moving -- not just stopped.
    let moving = transform.translation.y.is_finite(); // always true; kept the check explicit below via speed instead
    let _ = moving;
    if let Some(prev) = race.prev_checkpoint_distance {
        if dist > prev + 0.01 {
            race.wrong_way_timer += time.delta_secs();
        } else {
            race.wrong_way_timer = 0.0;
        }
    }
    race.prev_checkpoint_distance = Some(dist);

    if dist <= CHECKPOINT_RADIUS {
        let completed_lap = race.next_checkpoint == 0;
        if completed_lap {
            let now = time.elapsed_secs();
            let lap_time = now - race.lap_start;
            race.last_lap_time = Some(lap_time);
            let is_best = race.best_lap_time.map_or(true, |best| lap_time < best);
            if is_best {
                race.best_lap_time = Some(lap_time);
                ghost.best = std::mem::take(&mut ghost.current);
            } else {
                ghost.current.clear();
            }
            race.lap += 1;
            race.lap_start = now;
        }
        race.next_checkpoint = (race.next_checkpoint + 1) % checkpoints.len();
        race.prev_checkpoint_distance = None;
        race.wrong_way_timer = 0.0;

        if let Some(audio) = audio.as_ref() {
            commands.spawn((
                AudioPlayer(audio.lap_chime.clone()),
                one_shot(
                    PlaybackSettings::DESPAWN.with_volume(Volume::Linear(0.5)),
                    muted.0,
                ),
            ));
        }
    }

    // Highlight whichever checkpoint is next; the rest stay plain white.
    for (visual, material_handle) in &mut checkpoint_query {
        if let Some(mut material) = materials.get_mut(&material_handle.0) {
            let is_next = visual.0 == race.next_checkpoint;
            material.base_color = if is_next {
                Color::srgba(1.0, 0.85, 0.1, 0.55)
            } else {
                Color::srgba(0.9, 0.9, 0.9, 0.25)
            };
            material.emissive = if is_next {
                LinearRgba::rgb(0.6, 0.5, 0.0)
            } else {
                LinearRgba::rgb(0.2, 0.2, 0.2)
            };
        }
    }

    let showing_wrong_way = race.wrong_way_timer >= WRONG_WAY_TRIGGER_SECONDS;
    for mut color in &mut wrong_way_query {
        color.0 = color.0.with_alpha(if showing_wrong_way { 0.9 } else { 0.0 });
    }
}

/// Appends a (time-into-lap, position, rotation) sample every frame the race
/// is live, so a completed best lap (promoted in `race_checkpoints`) has a
/// full path for `update_ghost_visual` to replay.
fn record_ghost(
    time: Res<Time>,
    race: Res<Race>,
    mut ghost: ResMut<GhostRecording>,
    player_query: Query<&Transform, With<PlayerCar>>,
) {
    let Ok(transform) = player_query.single() else {
        return;
    };
    let t = time.elapsed_secs() - race.lap_start;
    ghost
        .current
        .push((t, transform.translation, transform.rotation));
}

/// Moves the translucent ghost car along `GhostRecording::best` to match how
/// far into the *current* lap the player now is -- so racing dead even with
/// your own best lap means driving side-by-side with the ghost the whole way
/// round.
fn update_ghost_visual(
    time: Res<Time>,
    race: Res<Race>,
    ghost: Res<GhostRecording>,
    mut ghost_query: Query<(&mut Transform, &mut Visibility), With<GhostCar>>,
) {
    let Ok((mut transform, mut visibility)) = ghost_query.single_mut() else {
        return;
    };
    if ghost.best.len() < 2 {
        *visibility = Visibility::Hidden;
        return;
    }
    *visibility = Visibility::Visible;

    let t = time.elapsed_secs() - race.lap_start;
    let samples = &ghost.best;
    // Linear scan for the bracketing pair -- a lap's worth of samples is a
    // few hundred entries at most, cheap enough once a frame.
    let idx = samples.partition_point(|(sample_t, _, _)| *sample_t < t);
    if idx == 0 {
        transform.translation = samples[0].1;
        transform.rotation = samples[0].2;
        return;
    }
    if idx >= samples.len() {
        let last = samples.last().unwrap();
        transform.translation = last.1;
        transform.rotation = last.2;
        return;
    }
    let (t0, p0, r0) = samples[idx - 1];
    let (t1, p1, r1) = samples[idx];
    let span = (t1 - t0).max(0.0001);
    let frac = ((t - t0) / span).clamp(0.0, 1.0);
    transform.translation = p0.lerp(p1, frac);
    transform.rotation = r0.slerp(r1, frac);
}

/// Charges while the player drifts and is actually sliding (meaningful
/// lateral speed under reduced grip), banks into `DriftScore::total` on a
/// clean release, and discards the charge instead if a single-frame speed
/// loss big enough to be a wall hit shows up first (see
/// `WALL_HIT_DECEL_THRESHOLD` -- there's no dependency on avian3d's own
/// collision events, just a before/after comparison).
fn drift_score_system(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut score: ResMut<DriftScore>,
    mut last_speed: Local<f32>,
    mut was_drifting: Local<bool>,
    player_query: Query<(&Transform, &LinearVelocity), With<PlayerCar>>,
) {
    let Ok((transform, lin_vel)) = player_query.single() else {
        return;
    };
    let speed = lin_vel.0.length();
    let drifting = keys.pressed(KeyCode::Space) && speed > 3.0;

    let hit_wall = *last_speed - speed > WALL_HIT_DECEL_THRESHOLD;
    *last_speed = speed;

    if hit_wall {
        score.current = 0.0;
        *was_drifting = false;
        return;
    }

    if drifting {
        let right: Vec3 = transform.right().into();
        let lateral = lin_vel.0.dot(right).abs();
        score.current += lateral * 2.0 * time.delta_secs();
        *was_drifting = true;
    } else if *was_drifting {
        // Clean release: bank whatever was charged.
        score.total += score.current;
        score.current = 0.0;
        *was_drifting = false;
    }
}

/// Slides the hazard barrier back and forth across its straight -- a plain
/// sine oscillation around `HazardSweeper::center_z`.
fn hazard_sweeper(time: Res<Time>, mut query: Query<(&HazardSweeper, &mut Transform)>) {
    for (hazard, mut transform) in &mut query {
        let offset = (time.elapsed_secs() * hazard.speed).sin() * hazard.amplitude;
        transform.translation.z = hazard.center_z + offset;
    }
}

/// Retargets the looping engine sound's pitch/volume to the player's own
/// speed every frame -- the clip itself is just a fixed loopable tone (see
/// `generate_audio.py`); all the "revving" comes from `AudioSink::set_speed`.
fn engine_audio(
    player_query: Query<(&Car, &LinearVelocity), With<PlayerCar>>,
    mut engine_query: Query<&mut AudioSink, With<EngineAudio>>,
) {
    let Ok((car, lin_vel)) = player_query.single() else {
        return;
    };
    let Ok(mut sink) = engine_query.single_mut() else {
        return;
    };
    let speed_factor = (lin_vel.0.length() / car.max_speed).clamp(0.0, 1.0);
    sink.set_speed(0.7 + speed_factor * 1.1);
    sink.set_volume(Volume::Linear(0.3 + speed_factor * 0.3));
}

/// Drives the drift-screech loop (pause/resume rather than despawn/respawn,
/// so it doesn't click or restart from the top every time the key taps) and
/// fires one-shot tire-chirp bursts on hard steering input.
fn tire_audio(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    audio: Option<Res<AudioAssets>>,
    muted: Res<Muted>,
    mut chirp_cooldown: Local<f32>,
    time: Res<Time>,
    player_query: Query<&LinearVelocity, With<PlayerCar>>,
    mut drift_sink_query: Query<&mut AudioSink, With<DriftAudio>>,
) {
    *chirp_cooldown -= time.delta_secs();
    let Ok(lin_vel) = player_query.single() else {
        return;
    };
    let speed = lin_vel.0.length();
    let drifting = keys.pressed(KeyCode::Space) && speed > 3.0;

    if let Ok(mut sink) = drift_sink_query.single_mut() {
        if drifting {
            sink.play();
            sink.set_volume(Volume::Linear(0.45));
        } else {
            sink.set_volume(Volume::Linear(0.0));
        }
    }

    let hard_steer =
        keys.any_pressed([KeyCode::KeyA, KeyCode::KeyD, KeyCode::ArrowLeft, KeyCode::ArrowRight]);
    if hard_steer && speed > 8.0 && *chirp_cooldown <= 0.0 {
        *chirp_cooldown = 0.6;
        if let Some(audio) = audio.as_ref() {
            commands.spawn((
                AudioPlayer(audio.tire_chirp.clone()),
                one_shot(
                    PlaybackSettings::DESPAWN.with_volume(Volume::Linear(0.3)),
                    muted.0,
                ),
            ));
        }
    }
}

/// Refreshes every HUD line (speed/lap/time/best/drift score) and the
/// WRONG WAY banner's text each frame from `Race`/`DriftScore`/the player's
/// own velocity. The banner's *visibility* (alpha) is set by
/// `race_checkpoints`; this only ever touches its text.
fn update_hud(
    race: Res<Race>,
    score: Res<DriftScore>,
    time: Res<Time>,
    player_query: Query<&LinearVelocity, With<PlayerCar>>,
    mut hud_query: Query<(&HudLine, &mut Text)>,
) {
    let speed = player_query
        .single()
        .map(|v| v.0.length())
        .unwrap_or(0.0);
    let lap_time = time.elapsed_secs() - race.lap_start;

    for (line, mut text) in &mut hud_query {
        match line {
            HudLine::Speed => text.0 = format!("SPEED {speed:.0}"),
            HudLine::Lap => text.0 = format!("LAP {}", race.lap),
            HudLine::LapTime => text.0 = format!("TIME {lap_time:.1}"),
            HudLine::BestLap => {
                text.0 = match race.best_lap_time {
                    Some(best) => format!("BEST {best:.1}"),
                    None => "BEST --".to_string(),
                }
            }
            HudLine::DriftScore => {
                text.0 = format!("DRIFT {:.0}", score.total + score.current)
            }
            HudLine::WrongWay => text.0 = "WRONG WAY".to_string(),
        }
    }
}

/// Cosmetic-only pitch tilt on the car's visual model: nose tilts up while
/// climbing (e.g. off a ramp), down while falling. Purely a rotation on the
/// glTF child's own local Transform -- the physics body (and its collider)
/// never moves, so this cannot destabilize the simulation the way physically
/// unlocking the body's own pitch repeatedly did.
fn car_visual_tilt(
    time: Res<Time>,
    car_query: Query<(&LinearVelocity, &Children), With<PlayerCar>>,
    mut visual_query: Query<(&mut Transform, &CarVisualBaseRotation), With<CarVisual>>,
) {
    let Ok((velocity, children)) = car_query.single() else {
        return;
    };
    // The player's own CarVisual child specifically -- not `visual_query.single()`,
    // which would error out as soon as any other car (AI, player 2) also has one.
    // Found read-only first (`contains`, no borrow conflict), then borrowed
    // mutably exactly once -- a `get_mut` call inside the search loop itself
    // can't satisfy the borrow checker across iterations.
    let Some(child) = children.iter().find(|&child| visual_query.contains(child)) else {
        return;
    };
    let Ok((mut visual_transform, base_rotation)) = visual_query.get_mut(child) else {
        return;
    };

    let horizontal_speed = Vec2::new(velocity.0.x, velocity.0.z).length();
    let tilt = velocity.0.y.atan2(horizontal_speed.max(1.0)).clamp(-0.5, 0.5);
    // Composed on top of the model's own rest orientation, not an absolute
    // target -- otherwise this overwrites (rather than tilts) whatever
    // orientation the model was spawned with.
    let target = base_rotation.0 * Quat::from_rotation_x(tilt);

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
    car_query: Query<(&Transform, &LinearVelocity), With<PlayerCar>>,
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

/// Drives every `FollowCamera` -- just the player 1 camera in single-player,
/// or both player cameras once `toggle_multiplayer` has spawned a second
/// one -- toward its own car, exactly as the single-camera version did.
fn camera_follow(
    player_query: Query<&Transform, (With<PlayerCar>, Without<FollowCamera>)>,
    player2_query: Query<&Transform, (With<Player2Car>, Without<FollowCamera>)>,
    mut camera_query: Query<(&FollowCamera, &mut Transform)>,
    time: Res<Time>,
) {
    for (follow, mut camera_transform) in &mut camera_query {
        let car_transform = if follow.follows_player2 {
            let Ok(t) = player2_query.single() else {
                continue;
            };
            t
        } else {
            let Ok(t) = player_query.single() else {
                continue;
            };
            t
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

/// `C` cycles the visible car model (currently the F1 car and the original
/// Kenney kart). Despawns the old `CarVisual` (which also tears down its
/// loaded glTF scene, since despawn cascades to children) and spawns a fresh
/// one for the new model via `spawn_car_visual`, so the two spawn paths can
/// never drift out of sync.
fn switch_car_model(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    asset_server: Res<AssetServer>,
    mut car_model: ResMut<CarModel>,
    mut car_body_query: Query<(Entity, &Children, &mut Car), With<CarBody>>,
    old_visual_query: Query<Entity, With<CarVisual>>,
) {
    if !keys.just_pressed(KeyCode::KeyC) {
        return;
    }
    let Ok((car_body, children, mut car)) = car_body_query.single_mut() else {
        return;
    };

    // Only CarBody's own child -- AI cars and (if active) player 2's car
    // each have their own CarVisual too, and must not be touched here.
    for child in children.iter() {
        if old_visual_query.get(child).is_ok() {
            commands.entity(child).despawn();
        }
    }

    *car_model = car_model.next();
    *car = car_model.stats();
    spawn_car_visual(&mut commands, car_body, &asset_server, *car_model);
}
