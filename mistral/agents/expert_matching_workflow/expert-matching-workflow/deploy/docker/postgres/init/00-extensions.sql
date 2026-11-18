-- Mounted as 020-extensions.sql so it runs after the image's bundled timescaledb
-- install/tune scripts. It only seeds a fresh local database: a migration that relies on an
-- extension must create it too, since autogenerate cannot see extensions.
-- `vector` (pgvector): vector columns and their similarity indexes.
-- `pg_textsearch`: the `bm25` access method for lexical indexes. It must also be in
-- shared_preload_libraries, which the pinned timescaledb-ha image already does; `CREATE EXTENSION`
-- alone is not enough on another image.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
