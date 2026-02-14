use schemars::schema::RootSchema;
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;

pub trait ModelClient<Deps> {
    fn complete(&self, prompt: &str, deps: &Deps) -> Result<String, Box<dyn Error + Send + Sync>>;
}

#[derive(Debug)]
pub enum AgentError {
    Model(String),
    Validation {
        attempts: usize,
        message: String,
    },
}

impl Display for AgentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(message) => write!(f, "model error: {message}"),
            Self::Validation { attempts, message } => {
                write!(f, "validation failed after {attempts} attempts: {message}")
            }
        }
    }
}

impl Error for AgentError {}

pub struct Agent<Deps, Output, Model> {
    model: Model,
    max_retries: usize,
    _marker: PhantomData<(Deps, Output)>,
}

impl<Deps, Output, Model> Agent<Deps, Output, Model>
where
    Output: DeserializeOwned + JsonSchema,
    Model: ModelClient<Deps>,
{
    pub fn new(model: Model) -> Self {
        Self {
            model,
            max_retries: 2,
            _marker: PhantomData,
        }
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn output_schema() -> RootSchema {
        schema_for!(Output)
    }

    pub fn run(&self, prompt: &str, deps: &Deps) -> Result<Output, AgentError> {
        let mut current_prompt = prompt.to_owned();
        let mut last_validation_error = None;

        for attempt in 0..=self.max_retries {
            let response = self
                .model
                .complete(&current_prompt, deps)
                .map_err(|err| AgentError::Model(err.to_string()))?;

            match serde_json::from_str::<Output>(&response) {
                Ok(output) => return Ok(output),
                Err(err) => {
                    last_validation_error = Some(err.to_string());

                    if attempt == self.max_retries {
                        break;
                    }

                    current_prompt = format!(
                        "{prompt}\n\nPrevious response did not match schema.\nValidation error: {err}\nReturn valid JSON only."
                    );
                }
            }
        }

        Err(AgentError::Validation {
            attempts: self.max_retries + 1,
            message: last_validation_error
                .unwrap_or_else(|| "unknown validation error".to_string()),
        })
    }
}

pub trait Tool {
    type Input: DeserializeOwned + JsonSchema;
    type Output;

    fn name(&self) -> &'static str;
    fn call(&self, input: Self::Input) -> Self::Output;

    fn input_schema() -> RootSchema {
        schema_for!(Self::Input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::cell::RefCell;

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    struct Answer {
        message: String,
    }

    #[derive(Default)]
    struct FakeModel {
        prompts: RefCell<Vec<String>>,
        responses: RefCell<Vec<String>>,
    }

    impl FakeModel {
        fn with_responses(responses: &[&str]) -> Self {
            Self {
                prompts: RefCell::new(Vec::new()),
                responses: RefCell::new(responses.iter().rev().map(|v| (*v).to_string()).collect()),
            }
        }
    }

    impl ModelClient<()> for FakeModel {
        fn complete(&self, prompt: &str, _deps: &()) -> Result<String, Box<dyn Error + Send + Sync>> {
            self.prompts.borrow_mut().push(prompt.to_string());
            Ok(self
                .responses
                .borrow_mut()
                .pop()
                .unwrap_or_else(|| "{\"message\":\"fallback\"}".to_string()))
        }
    }

    #[test]
    fn returns_typed_output() {
        let model = FakeModel::with_responses(&["{\"message\":\"hello\"}"]);
        let agent: Agent<(), Answer, _> = Agent::new(model);

        let output = agent.run("Say hello", &()).unwrap();

        assert_eq!(
            output,
            Answer {
                message: "hello".to_string()
            }
        );
    }

    #[test]
    fn retries_with_reflection_prompt_after_validation_error() {
        let model = FakeModel::with_responses(&["not-json", "{\"message\":\"fixed\"}"]);
        let agent: Agent<(), Answer, _> = Agent::new(model).with_max_retries(1);

        let output = agent.run("Need JSON", &()).unwrap();
        assert_eq!(output.message, "fixed");

        let prompts = agent.model.prompts.borrow();
        assert_eq!(prompts.len(), 2);
        assert!(prompts[1].contains("Validation error"));
        assert!(prompts[1].contains("Return valid JSON only."));
    }
}
