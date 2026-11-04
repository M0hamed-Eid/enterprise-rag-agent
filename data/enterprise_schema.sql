-- =====================================================================
-- NexusAgent Systems: Enterprise Metrics & Operations Database Schema
-- Compatible with SQLite and PostgreSQL
-- Purpose: Structured data grounding for LangChain SQL Database Chain
-- =====================================================================

-- ---------------------------------------------------------------------
-- Table: regional_sales
-- Description: Tracks quarterly revenue and target margins across 
--              global operating regions.
-- ---------------------------------------------------------------------
CREATE TABLE regional_sales (
    id INTEGER PRIMARY KEY,
    region VARCHAR(100) NOT NULL,
    q1_revenue DECIMAL(15, 2) NOT NULL,
    q2_revenue DECIMAL(15, 2) NOT NULL,
    target_margin DECIMAL(5, 2) NOT NULL
);

-- ---------------------------------------------------------------------
-- Table: employee_roster
-- Description: Central directory of NexusAgent employees, including 
--              internal data access clearance levels.
-- ---------------------------------------------------------------------
CREATE TABLE employee_roster (
    emp_id INTEGER PRIMARY KEY,
    department VARCHAR(100) NOT NULL,
    role VARCHAR(100) NOT NULL,
    clearance_level INTEGER NOT NULL
);

-- ---------------------------------------------------------------------
-- Table: client_contracts
-- Description: Details of ongoing, pending, and completed B2B contracts 
--              managed by NexusAgent Systems.
-- ---------------------------------------------------------------------
CREATE TABLE client_contracts (
    contract_id INTEGER PRIMARY KEY,
    client_name VARCHAR(150) NOT NULL,
    value_usd DECIMAL(15, 2) NOT NULL,
    status VARCHAR(50) NOT NULL
);

-- =====================================================================
-- Seed Data Insertion
-- =====================================================================

-- Seed regional_sales
INSERT INTO regional_sales (id, region, q1_revenue, q2_revenue, target_margin) VALUES
(1, 'North America', 1450000.00, 1550000.00, 22.50),
(2, 'EMEA', 980000.00, 1100000.00, 20.00),
(3, 'APAC', 1200000.00, 1350000.00, 25.00),
(4, 'LATAM', 450000.00, 520000.00, 18.50),
(5, 'Nordics', 350000.00, 410000.00, 21.00),
(6, 'Oceania', 280000.00, 310000.00, 19.50);

-- Seed employee_roster
INSERT INTO employee_roster (emp_id, department, role, clearance_level) VALUES
(101, 'Executive', 'Chief Executive Officer', 5),
(102, 'Executive', 'Chief Technology Officer', 5),
(103, 'AI Engineering', 'Lead AI Engineer', 4),
(104, 'AI Engineering', 'Mid-level AI Engineer', 3),
(105, 'AI Engineering', 'Data Scientist', 3),
(106, 'Sales', 'VP of Global Sales', 4),
(107, 'Sales', 'Enterprise Regional Director', 3),
(108, 'Sales', 'Senior Account Executive', 2),
(109, 'Human Resources', 'HR Director', 4),
(110, 'Human Resources', 'HR Business Partner', 3),
(111, 'Operations', 'Chief Operating Officer', 5),
(112, 'Operations', 'Operations Manager', 3),
(113, 'Legal', 'General Counsel', 4),
(114, 'IT Security', 'Chief Information Security Officer', 5),
(115, 'IT Security', 'Security Analyst', 3),
(116, 'Finance', 'Chief Financial Officer', 5),
(117, 'Finance', 'Financial Controller', 4);

-- Seed client_contracts
INSERT INTO client_contracts (contract_id, client_name, value_usd, status) VALUES
(1001, 'GlobalTech Industries', 2500000.00, 'Active'),
(1002, 'FinServe Banking Corp', 1200000.00, 'Pending Renewal'),
(1003, 'AeroSpace Dynamics', 3400000.00, 'Active'),
(1004, 'HealthPlus Network', 850000.00, 'Completed'),
(1005, 'Retail Giants LLC', 1750000.00, 'Active'),
(1006, 'EcoEnergy Solutions', 920000.00, 'In Negotiation'),
(1007, 'MediaCorp International', 1100000.00, 'Active'),
(1008, 'NextGen Auto', 2100000.00, 'Active'),
(1009, 'Urban Logistics', 670000.00, 'Completed'),
(1010, 'Quantum Computing Inc', 4200000.00, 'Pending Signature'),
(1011, 'AgriCorp', 540000.00, 'Active'),
(1012, 'EduTech Partners', 880000.00, 'In Negotiation'),
(1013, 'Apex Financial Group', 3100000.00, 'Active'),
(1014, 'SecureCloud Data Services', 1450000.00, 'Pending Renewal');