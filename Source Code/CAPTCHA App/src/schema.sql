-- Schema for the SQLite database to store user responses
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_agent TEXT,
    ip_address TEXT
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    row_index INTEGER NOT NULL,      -- Index of the dataframe row
    sentence TEXT NOT NULL,          -- The sentence shown to user
    selected_sdg INTEGER,            -- SDG number selected by user (NULL if "None")
    relevant_sdgs TEXT,              -- JSON string of relevant SDGs
    sdg_totals TEXT,                 -- JSON string of SDG totals
    max_total REAL,                  -- Max total value
    is_correct BOOLEAN,              -- Whether the response was correct
    response_type TEXT,              -- "correct", "not_sure", "incorrect" 
    streak INTEGER,                  -- User's streak at time of response
    response_time_ms INTEGER,        -- Time taken to respond in milliseconds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_responses_session_id ON responses(session_id);
CREATE INDEX IF NOT EXISTS idx_responses_created_at ON responses(created_at);
