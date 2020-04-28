use serde::{Serialize, Deserialize};
use std::error::Error;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::{BufRead, BufReader};
use aho_corasick::AhoCorasick;

pub struct Config {
    pub infile: String,
    pub outfile: String,
    pub query: String,
}

#[derive(Deserialize, Debug)]
pub struct Input {
    id: Option<String>,
    created_at: Option<String>,
    user_id: Option<String>,
    text: Option<String>,
    extended_text: Option<String>,
    lang: Option<String>,
}

impl Input {
    fn full_text(&self) -> Option<String> {
        match &self.extended_text {
            None => self.text.clone(),
            Some(x) => Some(x.to_string()),
        }
    }
}

#[derive(Serialize, Debug)]
pub struct Output {
    id: Option<String>,
    created_at: Option<String>,
    user_id: Option<String>,
    full_text: Option<String>,
    lang: Option<String>,
}

impl From<Input> for Output {
    fn from(item: Input) -> Output {
        Output {
            full_text: item.full_text(),
            id: item.id,
            created_at: item.created_at,
            user_id: item.user_id,
            lang: item.lang,
        }
    }
}

impl Config {
    pub fn new(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 4 {
            return Err("not enough arguments");
        }
        let infile = args[1].clone();
        let outfile = args[2].clone();
        let query = args[3].clone();

        Ok(Config { infile, outfile, query })
    }
}

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let queryfile = File::open(config.query)?;
    let queryreader = BufReader::new(queryfile);
    let patterns: Vec<String> = queryreader
        .lines()
        .map(|l| l.expect("Could not parse line"))
        .collect();
    let ac = AhoCorasick::new(patterns);

    let infile = File::open(config.infile)?;
    let reader = BufReader::new(infile);
    let mut outfile = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(config.outfile)
        .unwrap();

    for line in reader.lines() {
        let v: Input = serde_json::from_str(&line.unwrap())?;
        if (&v.lang.as_ref().unwrap().to_string() == "en") && (ac.is_match(&v.text.as_ref().unwrap())) {
            let output = Output::from(v);
            let j = serde_json::to_string(&output)?;
            if let Err(e) = writeln!(outfile, "{}", &j) {
                eprintln!("Couldn't write to file: {}", e);
            }
            //println!("{}", j);
        }
    }
    Ok(())
}
