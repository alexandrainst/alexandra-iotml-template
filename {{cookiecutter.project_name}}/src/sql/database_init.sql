/* SQL script for initializing a standard "observations"
hypertable for IoT / time series project.The table has 
three columns:

- "time", which holds the timestamp of a measurement
- "uuid", which uniquely identifies a sensor
- "data", a JSONB field where all measurements are stored
*/
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE TABLE observations(
	id SERIAL NOT NULL,
	time TIMESTAMP WITH TIME ZONE NOT NULL,
	uuid TEXT,
	data JSONB
);

/* Example command to convert postgresql table into a timescale hypertable */
SELECT create_hypertable('observations', 'time', if_not_exists => TRUE);

/* Create an index on the uuid */
CREATE INDEX IF NOT EXISTS index_sensor_id ON observations("uuid");

/* Create a GIN index to allow indexing of JSONB data */
CREATE INDEX IF NOT EXISTS datagin ON OBSERVATIONS USING gin (data);
