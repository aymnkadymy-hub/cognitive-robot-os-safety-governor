//! # behavior-fsm — behavior/decision layer (finite state machine)
//!
//! The simplest practical planning layer for a robot: `Idle → Search → Approach → Act → Recover`.
//! Takes a **perception summary** (is the target visible/near? is the situation safe?) and outputs
//! a **state + high-level intent** that feeds the safety guard/controller. Simpler and more robust
//! than an isolated LLM, and `no_std`-ready for seL4. (Priority 4.)

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

/// Behavior state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Idle,     // idle — no target
    Search,   // searching — target detected, exploring toward it
    Approach, // approaching the target
    Act,      // interacting (target within reach)
    Recover,  // recovery — unsafe situation
}

/// High-level intent output (translated to commands for the guard/controller).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Intent {
    Hold,       // stop
    Scan,       // rotate/explore
    MoveToward, // move toward target
    Manipulate, // execute task
    BackOff,    // safe retreat
}

/// Perception summary driving state transitions (derived from world memory/sensors).
#[derive(Clone, Copy, Debug, Default)]
pub struct Percept {
    pub target_visible: bool,   // is a target visible?
    pub target_near: bool,      // is it within reach?
    pub unsafe_situation: bool, // is the situation dangerous? (overrides everything)
}

/// Behavior finite state machine.
#[derive(Clone, Copy, Debug)]
pub struct BehaviorFsm {
    state: State,
}

impl BehaviorFsm {
    pub const fn new() -> Self {
        Self { state: State::Idle }
    }

    pub fn state(&self) -> State {
        self.state
    }

    /// Single decision step: updates state from perception and returns it. Safety takes priority.
    pub fn step(&mut self, p: Percept) -> State {
        self.state = if p.unsafe_situation {
            State::Recover
        } else {
            match self.state {
                State::Idle => {
                    if p.target_visible {
                        State::Search
                    } else {
                        State::Idle
                    }
                }
                State::Search => {
                    if !p.target_visible {
                        State::Idle
                    } else if p.target_near {
                        State::Approach
                    } else {
                        State::Search
                    }
                }
                State::Approach => {
                    if !p.target_visible {
                        State::Search
                    } else if p.target_near {
                        State::Act
                    } else {
                        State::Approach
                    }
                }
                State::Act => {
                    if !p.target_near {
                        State::Approach
                    } else {
                        State::Act
                    }
                }
                // Recovery ends when the danger is gone.
                State::Recover => State::Idle,
            }
        };
        self.state
    }

    /// High-level intent for the current state.
    pub fn intent(&self) -> Intent {
        match self.state {
            State::Idle => Intent::Hold,
            State::Search => Intent::Scan,
            State::Approach => Intent::MoveToward,
            State::Act => Intent::Manipulate,
            State::Recover => Intent::BackOff,
        }
    }
}

impl Default for BehaviorFsm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(vis: bool, near: bool, danger: bool) -> Percept {
        Percept {
            target_visible: vis,
            target_near: near,
            unsafe_situation: danger,
        }
    }

    #[test]
    fn full_mission_cycle() {
        let mut f = BehaviorFsm::new();
        assert_eq!(f.state(), State::Idle);
        assert_eq!(f.step(p(true, false, false)), State::Search); // target detected
        assert_eq!(f.step(p(true, true, false)), State::Approach); // approaching
        assert_eq!(f.step(p(true, true, false)), State::Act); // reached → act
        assert_eq!(f.intent(), Intent::Manipulate);
    }

    #[test]
    fn safety_preempts_everything() {
        let mut f = BehaviorFsm::new();
        f.step(p(true, true, false)); // Search
        f.step(p(true, true, false)); // Approach
                                      // Sudden danger → immediate recovery regardless of current state.
        assert_eq!(f.step(p(true, true, true)), State::Recover);
        assert_eq!(f.intent(), Intent::BackOff);
    }

    #[test]
    fn loses_target_falls_back() {
        let mut f = BehaviorFsm::new();
        f.step(p(true, false, false)); // Search
        assert_eq!(f.step(p(false, false, false)), State::Idle); // target lost
    }

    #[test]
    fn recovers_to_idle_when_safe() {
        let mut f = BehaviorFsm::new();
        f.step(p(true, true, true)); // Recover
        assert_eq!(f.step(p(false, false, false)), State::Idle);
    }
}
