-- ============================================================
-- LuminaData Enterprise v6.0 — Database Initialization
-- Merged: original 25-row seed + v7 governance columns
-- Compatible with: mcp_server.py, orchestration.py, app.py
-- ============================================================

-- 1. Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Citizens table
CREATE TABLE IF NOT EXISTS citizens (
    id                SERIAL PRIMARY KEY,
    national_id       VARCHAR(10) UNIQUE,
    full_name         VARCHAR(100) NOT NULL,
    date_of_birth     DATE,
    gender            VARCHAR(10),
    city              VARCHAR(50),
    region            VARCHAR(50),
    phone_number      VARCHAR(15),
    email             VARCHAR(100),
    id_issue_date     DATE,
    id_expiry_date    DATE,
    marital_status    VARCHAR(20),
    education_level   VARCHAR(30),
    employment_status VARCHAR(30),
    registration_date TIMESTAMP DEFAULT NOW(),
    age_group         VARCHAR(20),
    -- Governance columns (from v7 upgrade)
    is_verified       BOOLEAN      DEFAULT FALSE,
    compliance_status VARCHAR(20)  DEFAULT 'Pending',
    data_source       VARCHAR(50)  DEFAULT 'Manual',
    last_updated_by   VARCHAR(50)
);

-- 3. Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id             SERIAL PRIMARY KEY,
    user_name      VARCHAR(50)  NOT NULL,
    user_role      VARCHAR(20)  NOT NULL DEFAULT 'Admin',
    agent_name     VARCHAR(50),
    action_type    VARCHAR(50)  NOT NULL,
    original_issue TEXT,
    reasoning      TEXT,
    executed_sql   TEXT         NOT NULL DEFAULT '',
    mcp_tool_used  VARCHAR(100),
    timestamp      TIMESTAMP    NOT NULL DEFAULT NOW()
);

-- 4. Decision memory table
-- Column names match mcp_server.py exactly: approved_sql, approved_by, timestamp
CREATE TABLE IF NOT EXISTS decision_memory (
    id                SERIAL PRIMARY KEY,
    issue_description TEXT        NOT NULL,
    embedding         vector(384),
    approved_sql      TEXT        NOT NULL,
    approved_by       VARCHAR(50) NOT NULL,
    timestamp         TIMESTAMP   NOT NULL DEFAULT NOW()
);

-- 5. Indexes
CREATE INDEX IF NOT EXISTS idx_citizens_city        ON citizens(city);
CREATE INDEX IF NOT EXISTS idx_citizens_region      ON citizens(region);
CREATE INDEX IF NOT EXISTS idx_citizens_age_group   ON citizens(age_group);
CREATE INDEX IF NOT EXISTS idx_citizens_employment  ON citizens(employment_status);
CREATE INDEX IF NOT EXISTS idx_citizens_national_id ON citizens(national_id);
CREATE INDEX IF NOT EXISTS idx_citizens_compliance  ON citizens(compliance_status);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp      ON audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user           ON audit_logs(user_name);
CREATE INDEX IF NOT EXISTS idx_audit_agent          ON audit_logs(agent_name);
CREATE INDEX IF NOT EXISTS idx_memory_embedding
    ON decision_memory USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

-- 6. Seed data — 25 citizens with intentional DQ flaws
INSERT INTO citizens
    (national_id, full_name, date_of_birth, gender, city, region,
     phone_number, email, id_issue_date, id_expiry_date,
     marital_status, education_level, employment_status, age_group,
     compliance_status, data_source)
VALUES
-- Clean records
('1023456789','Ahmed Al-Rashidi',   '1985-03-15','Male',  'Riyadh','Riyadh',  '0501234567','ahmed.rashidi@email.com', '2015-01-10','2025-01-10','Married','Bachelor',   'Employed',   'Adult', 'Compliant',     'Absher'),
('1034567890','Fatima Al-Zahrani',  '1990-07-22','Female','Jeddah','Makkah',  '0512345678','fatima.z@email.com',      '2018-05-20','2028-05-20','Single', 'Master',      'Employed',   'Youth', 'Compliant',     'Absher'),
-- Missing phone and email
('1045678901','Mohammed Al-Otaibi', '2000-11-30','Male',  'Mecca', 'Makkah',  NULL,        NULL,                      '2020-09-15','2030-09-15','Single', 'High School', 'Student',    'Youth', 'Pending',       'Manual'),
-- Expired ID
('1056789012','Noura Al-Shehri',    '1978-04-18','Female','Riyadh','Riyadh',  '0534567890','noura.s@email.com',       '2010-03-01','2020-03-01','Married','Bachelor',    'Employed',   'Adult', 'Non-Compliant', 'Manual'),
-- Expired ID
('1067890123','Khalid Al-Harbi',    '1965-09-05','Male',  'Dammam','Eastern', '0545678901',NULL,                      '2005-06-10','2015-06-10','Married','Diploma',     'Retired',    'Senior','Non-Compliant', 'Batch'),
-- Missing national_id
(NULL,        'Sara Al-Ghamdi',     '1995-12-10','Female','Riyadh','Riyadh',  '0556789012','sara.g@email.com',        '2022-01-05','2032-01-05','Single', 'Bachelor',    'Employed',   'Youth', 'Pending',       'Manual'),
-- Date logic error: expiry BEFORE issue
('1089012345','Abdullah Al-Qahtani','1988-06-25','Male',  'Medina','Madinah', '0567890123','abdullah.q@email.com',    '2025-08-10','2020-08-10','Married','Master',      'Employed',   'Adult', 'Non-Compliant', 'Manual'),
-- Clean
('1090123456','Reem Al-Dosari',     '2005-02-14','Female','Dammam','Eastern', '0578901234',NULL,                      '2021-11-20','2031-11-20','Single', 'High School', 'Student',    'Youth', 'Pending',       'Absher'),
-- Missing national_id
(NULL,        'Omar Al-Mutairi',    '1972-08-08','Male',  'Riyadh','Riyadh',  NULL,        'omar.m@email.com',        '2008-04-15','2018-04-15','Married','Bachelor',    'Employed',   'Adult', 'Pending',       'Manual'),
-- Expired ID
('1012345670','Maha Al-Aqil',       '1982-10-30','Female','Jeddah','Makkah',  '0590123456','maha.a@email.com',        '2012-07-22','2022-07-22','Divorced','Doctorate',  'Employed',   'Adult', 'Non-Compliant', 'Absher'),
-- Clean
('1023456780','Faisal Al-Shamrani', '1998-05-17','Male',  'Abha',  'Asir',    '0501111222','faisal.sh@email.com',     '2019-03-10','2029-03-10','Single', 'Bachelor',    'Unemployed', 'Youth', 'Compliant',     'Absher'),
-- Expired ID
('1034567891','Hessa Al-Subaie',    '1969-01-28','Female','Riyadh','Riyadh',  '0512222333',NULL,                      '2007-09-05','2017-09-05','Married','Diploma',     'Employed',   'Senior','Non-Compliant', 'Manual'),
-- Clean
('1045678902','Turki Al-Anzi',      '2003-07-11','Male',  'Tabuk', 'Tabuk',   '0523333444','turki.a@email.com',       '2021-12-01','2031-12-01','Single', 'High School', 'Student',    'Youth', 'Compliant',     'Absher'),
-- Missing national_id + phone + email
(NULL,        'Dalal Al-Bishi',     '1991-09-19','Female','Jizan', 'Jizan',   NULL,        NULL,                      '2017-06-20','2027-06-20','Married','Bachelor',    'Employed',   'Youth', 'Pending',       'Manual'),
-- Expired ID
('1067890124','Nasser Al-Yami',     '1960-03-03','Male',  'Najran','Najran',  '0545555666','nasser.y@email.com',      '2003-02-14','2013-02-14','Married','Elementary',  'Retired',    'Senior','Non-Compliant', 'Batch'),
-- Expired ID
('1078901235','Wafa Al-Maliki',     '1986-11-07','Female','Riyadh','Riyadh',  '0556666777','wafa.m@email.com',        '2014-08-30','2024-08-30','Single', 'Master',      'Employed',   'Adult', 'Non-Compliant', 'Absher'),
-- Clean
('1089012346','Saad Al-Ruwaili',    '1993-04-24','Male',  'Al Jouf','Al Jawf','0567777888',NULL,                      '2020-05-15','2030-05-15','Single', 'Bachelor',    'Employed',   'Youth', 'Pending',       'Manual'),
-- Expired ID
('1090123457','Amal Al-Asiri',      '1975-12-31','Female','Abha',  'Asir',    '0578888999','amal.as@email.com',       '2009-10-10','2019-10-10','Married','Bachelor',    'Employed',   'Adult', 'Non-Compliant', 'Manual'),
-- Clean
('1001234568','Yazeed Al-Silmi',    '2001-08-16','Male',  'Riyadh','Riyadh',  '0589999000','yazeed.s@email.com',      '2022-07-01','2032-07-01','Single', 'Bachelor',    'Student',    'Youth', 'Compliant',     'Absher'),
-- Expired ID
('1012345671','Lujain Al-Tamimi',   '1984-05-29','Female','Dammam','Eastern', '0590000111','lujain.t@email.com',      '2013-04-18','2023-04-18','Married','Master',      'Employed',   'Adult', 'Non-Compliant', 'Absher'),
-- Expired ID
('1023456781','Bandar Al-Suwailem', '1970-02-20','Male',  'Medina','Madinah', '0501010101',NULL,                      '2006-11-25','2016-11-25','Married','Diploma',     'Employed',   'Senior','Non-Compliant', 'Batch'),
-- Missing national_id
(NULL,        'Ghada Al-Qurashi',   '1997-06-06','Female','Jeddah','Makkah',  '0512020202','ghada.q@email.com',       '2020-02-28','2030-02-28','Single', 'Bachelor',    'Unemployed', 'Youth', 'Pending',       'Manual'),
-- Expired ID
('1045678903','Majed Al-Zahrani',   '1963-10-14','Male',  'Riyadh','Riyadh',  '0523030303','majed.z@email.com',       '2004-07-07','2014-07-07','Married','Bachelor',    'Retired',    'Senior','Non-Compliant', 'Manual'),
-- Clean
('1056789013','Tahani Al-Ghamdiy',  '1988-03-22','Female','Mecca', 'Makkah',  '0534040404','tahani.g@email.com',      '2016-09-12','2026-09-12','Divorced','Doctorate',  'Employed',   'Adult', 'Compliant',     'Absher'),
-- Clean
('1067890125','Nawaf Al-Shammari',  '2004-01-09','Male',  'Hail',  'Hail',    '0545050505',NULL,                      '2023-03-20','2033-03-20','Single', 'High School', 'Student',    'Youth', 'Compliant',     'Absher');

-- 7. Seed decision memory (one example entry, no embedding — will be populated by agents)
INSERT INTO decision_memory (issue_description, approved_sql, approved_by)
VALUES (
    'Expiry date is before issue date — date logic error',
    'UPDATE citizens SET compliance_status = ''Non-Compliant'' WHERE id_expiry_date < id_issue_date;',
    'system'
);