-- JAVIS PostgreSQL Initialization Script
-- This script runs when the PostgreSQL container is first created

-- Create MLflow database
CREATE DATABASE mlflow;

-- Grant all privileges on mlflow database
GRANT ALL PRIVILEGES ON DATABASE mlflow TO javis;

-- Connect to javis database and create extensions
\c javis

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schema for JAVIS tables
CREATE SCHEMA IF NOT EXISTS javis;

-- Conversations table
CREATE TABLE IF NOT EXISTS javis.conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL UNIQUE,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    feedback VARCHAR(50),
    tags TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Conversation turns table
CREATE TABLE IF NOT EXISTS javis.conversation_turns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES javis.conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    turn_index INTEGER NOT NULL
);

-- Model versions table
CREATE TABLE IF NOT EXISTS javis.model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending',
    is_active BOOLEAN DEFAULT FALSE,
    adapter_path TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Training runs table
CREATE TABLE IF NOT EXISTS javis.training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id VARCHAR(255) NOT NULL UNIQUE,
    version VARCHAR(100) REFERENCES javis.model_versions(version),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'running',
    config JSONB DEFAULT '{}'::jsonb,
    metrics JSONB DEFAULT '{}'::jsonb,
    error_message TEXT
);

-- A/B tests table
CREATE TABLE IF NOT EXISTS javis.ab_tests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_id VARCHAR(255) NOT NULL UNIQUE,
    version_a VARCHAR(100) NOT NULL,
    version_b VARCHAR(100) NOT NULL,
    traffic_split DECIMAL(3,2) DEFAULT 0.50,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'active',
    description TEXT,
    results JSONB DEFAULT '{}'::jsonb
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS javis.metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL NOT NULL,
    version VARCHAR(100),
    labels JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON javis.conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_started_at ON javis.conversations(started_at);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation_id ON javis.conversation_turns(conversation_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_status ON javis.model_versions(status);
CREATE INDEX IF NOT EXISTS idx_model_versions_is_active ON javis.model_versions(is_active);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON javis.training_runs(status);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON javis.metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON javis.metrics(metric_name);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to conversations table
DROP TRIGGER IF EXISTS update_conversations_updated_at ON javis.conversations;
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON javis.conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT USAGE ON SCHEMA javis TO javis;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA javis TO javis;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA javis TO javis;

-- Connect to mlflow database and set up
\c mlflow

-- Grant privileges
GRANT ALL PRIVILEGES ON SCHEMA public TO javis;

RAISE NOTICE 'JAVIS database initialization complete!';
