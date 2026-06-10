//! SIL: the robot **perceives its 3D world and knows its surroundings** — on the OS architecture.
//!
//! It navigates a Minecraft world (table/chair/tree at real positions from `minecraft_world.xml`),
//! "sees" everything within sensor range, stores it in **world-memory** (embedding + pose +
//! class + time), then queries: what is around me? what is nearest? where is the nearest chair?
//! what is this object?
//!
//! The encoder here is simplified (class one-hot + noise) — in the full system it is produced
//! by the perception layer's vision module.
//! `world-memory` is `no_std` and runs on seL4.

use world_memory::WorldMemory;

const DIM: usize = 8;
const SENSOR_RANGE: f32 = 1.8; // sensing/vision range

const CLASSES: [&str; 3] = ["table", "chair", "tree"];

/// World objects: (class, x, y) — matching minecraft_world.xml.
const OBJECTS: [(u32, f32, f32); 3] = [
    (0, 2.0, -1.2), // table
    (1, 1.0, -1.3), // chair
    (2, 2.4, 1.6),  // tree
];

/// Simplified encoder: a distinct embedding per class + deterministic noise (simulates
/// realistic imperfect perception).
fn encode(class: u32, t: u32) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    v[(class as usize) % DIM] = 1.0;
    v[((class as usize) + 3) % DIM] = 0.6;
    let n = ((t as f32) * 0.13).sin() * 0.08; // small noise
    for x in v.iter_mut() {
        *x += n;
    }
    v
}

fn main() {
    let mut world = WorldMemory::<64, DIM>::new();

    // Robot waypoint path.
    let path = [(0.0f32, 0.0f32), (1.2, -1.0), (2.0, -0.5), (2.4, 0.8)];
    let mut t = 0u32;
    println!("[world] robot explores its 3D world, perceiving objects within {SENSOR_RANGE}m:");
    for (wp, &(rx, ry)) in path.iter().enumerate() {
        for &(class, ox, oy) in OBJECTS.iter() {
            let (dx, dy) = (ox - rx, oy - ry);
            let dist = (dx * dx + dy * dy).sqrt();
            if dist <= SENSOR_RANGE {
                world.observe(&encode(class, t), [ox, oy, 0.0], class, t);
                println!(
                    "[world]   waypoint {} ({:.1},{:.1}): perceived '{}' at ({:.1},{:.1}), {:.1}m away",
                    wp, rx, ry, CLASSES[class as usize], ox, oy, dist
                );
                t += 1;
            }
        }
    }
    println!(
        "[world] world-memory now holds {} perceptions (a map of the surroundings)\n",
        world.len()
    );

    // ===== The robot queries its world model =====
    let (rx, ry) = *path.last().unwrap();
    println!(
        "[world] robot is now at ({rx:.1},{ry:.1}). What does it know about its surroundings?"
    );

    // 1) What is around me?
    let around = world.count_within(rx, ry, 2.0);
    println!("[world] Q: what is around me (<=2.0m)?  -> {around} objects");

    // 2) What is the nearest object, and what is it?
    if let Some(h) = world.nearest(rx, ry) {
        let e = world.get(h.index).unwrap();
        println!(
            "[world] Q: nearest object?  -> '{}' at ({:.1},{:.1})",
            CLASSES[e.class as usize], e.pose[0], e.pose[1]
        );
    }

    // 3) Where is the nearest chair?
    if let Some(h) = world.nearest_of_class(1, rx, ry) {
        let e = world.get(h.index).unwrap();
        println!(
            "[world] Q: where is the nearest chair?  -> at ({:.1},{:.1})",
            e.pose[0], e.pose[1]
        );
    }

    // 4) "What is this?" — perceives a new object and identifies it by similarity.
    let unknown = encode(2, 999); // something that resembles the tree (new perception)
    if let Some(h) = world.recall_similar(&unknown) {
        let e = world.get(h.index).unwrap();
        println!(
            "[world] Q: what is this thing I see?  -> recognized as '{}' (similarity {:.2})",
            CLASSES[e.class as usize], h.score
        );
    }

    println!("\n[world] PASS: the OS gives the robot a queryable model of its world (perception layer 3).");
}
