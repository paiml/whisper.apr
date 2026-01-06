use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Worker, MessageEvent};
use crate::worker::{WorkerCommand, WorkerResult};

pub struct WorkerBridge {
    worker: Worker,
    on_result: Option<Closure<dyn Fn(MessageEvent)>>,
}

impl WorkerBridge {
    /// Create a new worker bridge.
    ///
    /// # Errors
    ///
    /// Returns error if Worker creation fails.
    pub fn new() -> Result<Self, JsValue> {
        let worker = Worker::new("./worker.js")?;

        let bridge = Self {
            worker,
            on_result: None,
        };

        Ok(bridge)
    }

    pub fn set_on_result<F>(&mut self, callback: F)
    where
        F: Fn(WorkerResult) + 'static,
    {
        let callback = Rc::new(callback);
        let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
            let data = event.data();
            match serde_wasm_bindgen::from_value::<WorkerResult>(data) {
                Ok(result) => callback(result),
                Err(e) => {
                    web_sys::console::error_1(&format!("Failed to deserialize worker message: {e:?}").into());
                }
            }
        }) as Box<dyn Fn(MessageEvent)>);

        self.worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        self.on_result = Some(on_message);
    }

    /// Post a command to the worker.
    ///
    /// # Errors
    ///
    /// Returns error if serialization or message posting fails.
    pub fn post_command(&self, command: &WorkerCommand) -> Result<(), JsValue> {
        let value = serde_wasm_bindgen::to_value(command)?;
        self.worker.post_message(&value)
    }
}
