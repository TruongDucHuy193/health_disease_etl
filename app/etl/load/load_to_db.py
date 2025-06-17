import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from pathlib import Path
import logging
from typing import Dict, Any
import os
from urllib.parse import quote_plus

def load_to_database(
    transformed_data: Dict[str, pd.DataFrame], 
    db_config: Dict[str, str] = None,
    table_prefix: str = "heart_disease"
) -> Dict[str, Any]:
    """Load transformed DataFrames to PostgreSQL database"""
    
    # Default PostgreSQL configuration
    if db_config is None:
        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'heart_disease_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '2310'),
            'schema': os.getenv('POSTGRES_SCHEMA', 'heart_disease')
        }
    
    stats = {
        "tables_created": 0,
        "total_rows_loaded": 0,
        "tables": {},
        "database_info": {
            "host": db_config['host'],
            "database": db_config['database'],
            "schema": db_config['schema']
        }
    }
    
    try:
        # Create PostgreSQL connection string
        password = quote_plus(db_config['password'])
        connection_string = (
            f"postgresql://{db_config['username']}:{password}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        # Create SQLAlchemy engine
        engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600
        )
        
        print(f"🔗 Connecting to PostgreSQL at {db_config['host']}:{db_config['port']}")
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Connected to PostgreSQL: {version[:50]}...")
        
        # Create schema if it doesn't exist
        with engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {db_config['schema']}"))
            conn.commit()
            print(f"📁 Schema '{db_config['schema']}' ready")
        
        # Load each DataFrame to a separate table
        for filename, df in transformed_data.items():
            print(f"\n💾 Loading {filename}...")
            
            # Clean filename for table name
            clean_filename = filename.replace('.csv', '').replace('processed_', '').replace('-', '_').replace(' ', '_').lower()
            table_name = f"{table_prefix}_{clean_filename}"
            
            # Prepare DataFrame for PostgreSQL
            df_clean = prepare_dataframe_for_postgres(df)
            
            # Load DataFrame to database
            rows_loaded = len(df_clean)
            df_clean.to_sql(
                name=table_name,
                con=engine,
                schema=db_config['schema'],
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            stats["tables_created"] += 1
            stats["total_rows_loaded"] += rows_loaded
            stats["tables"][table_name] = rows_loaded
            
            print(f"   ✅ Loaded {rows_loaded} rows to table '{db_config['schema']}.{table_name}'")
        
        # Create merged table if we have multiple regional datasets
        if len(stats["tables"]) >= 2:
            print(f"\n🔗 Creating merged table...")
            merged_table_name = f"{table_prefix}_merged"
            
            create_merged_table(engine, db_config['schema'], list(stats["tables"].keys()), merged_table_name)
            
            # Get count of merged table
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {db_config['schema']}.{merged_table_name}"))
                merged_count = result.fetchone()[0]
                
            stats["tables"][merged_table_name] = merged_count
            stats["tables_created"] += 1
            
            print(f"   ✅ Created merged table '{db_config['schema']}.{merged_table_name}' with {merged_count} rows")
        
        # Close engine
        engine.dispose()
        
        print(f"\n🎉 Database operations completed!")
        print(f"   📊 Tables created: {stats['tables_created']}")
        print(f"   📈 Total rows loaded: {stats['total_rows_loaded']}")
        
    except Exception as e:
        print(f"❌ Database load failed: {e}")
        logging.error(f"Database load failed: {e}")
        raise
    
    return stats

def prepare_dataframe_for_postgres(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for PostgreSQL insertion"""
    df_clean = df.copy()
    
    # Handle missing values and infinities
    import numpy as np
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)
    
    # Ensure proper data types for PostgreSQL
    # Integer columns
    integer_columns = [
        'sex', 'cp', 'exang', 'fbs', 'restecg', 'slope', 'ca', 'thal',
        'dig', 'prop', 'nitr', 'pro', 'diuretic', 'smoke', 'htn', 'dm', 
        'famhist', 'xhypo'
    ]
    
    for col in integer_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype('int32')
    
    # Float columns
    float_columns = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'restef', 'restwm',
        'exeref', 'exerwm', 'cigs', 'lmt', 'ladprox', 'laddist', 'diag',
        'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist'
    ]
    
    for col in float_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0).astype('float32')
    
    return df_clean

def create_merged_table(engine, schema_name, table_names, merged_table_name):
    """Create a merged table from multiple tables"""
    
    try:
        with engine.connect() as conn:
            # Drop existing merged table if exists
            conn.execute(text(f"DROP TABLE IF EXISTS {schema_name}.{merged_table_name}"))
            
            # Create UNION query to merge all tables
            select_queries = []
            for table_name in table_names:
                # Extract region name from table name
                region = table_name.replace('heart_disease_', '').replace('_', ' ').title()
                select_queries.append(f"SELECT *, '{region}' as source_region FROM {schema_name}.{table_name}")
            
            union_query = " UNION ALL ".join(select_queries)
            create_merged_sql = f"CREATE TABLE {schema_name}.{merged_table_name} AS {union_query}"
            
            conn.execute(text(create_merged_sql))
            conn.commit()
            
            print(f"   🔗 Merged {len(table_names)} tables into '{merged_table_name}'")
            
    except Exception as e:
        print(f"   ⚠️ Could not create merged table: {e}")

def create_indexes(db_config: Dict[str, str] = None) -> None:
    """Create indexes for better query performance"""
    
    if db_config is None:
        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'heart_disease_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'your_password'),
            'schema': os.getenv('POSTGRES_SCHEMA', 'heart_disease')
        }
    
    try:
        # Create connection string
        password = quote_plus(db_config['password'])
        connection_string = (
            f"postgresql://{db_config['username']}:{password}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        engine = create_engine(connection_string)
        
        print("🔧 Creating database indexes...")
        
        with engine.connect() as conn:
            # Common indexes for heart disease analysis
            indexes = [
                f"CREATE INDEX IF NOT EXISTS idx_age ON {db_config['schema']}.heart_disease_merged(age)",
                f"CREATE INDEX IF NOT EXISTS idx_sex ON {db_config['schema']}.heart_disease_merged(sex)",
                f"CREATE INDEX IF NOT EXISTS idx_chest_pain ON {db_config['schema']}.heart_disease_merged(cp)",
                f"CREATE INDEX IF NOT EXISTS idx_source ON {db_config['schema']}.heart_disease_merged(source_region)",
                f"CREATE INDEX IF NOT EXISTS idx_age_sex ON {db_config['schema']}.heart_disease_merged(age, sex)"
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                except Exception as e:
                    print(f"   ⚠️ Could not create index: {e}")
            
            conn.commit()
        
        engine.dispose()
        print("✅ Database indexes created successfully")
        
    except Exception as e:
        print(f"❌ Index creation failed: {e}")
        logging.error(f"Index creation failed: {e}")

def test_database_connection(db_config: Dict[str, str] = None) -> bool:
    """Test PostgreSQL database connection"""
    
    if db_config is None:
        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'heart_disease_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'your_password'),
            'schema': os.getenv('POSTGRES_SCHEMA', 'heart_disease')
        }
    
    try:
        password = quote_plus(db_config['password'])
        connection_string = (
            f"postgresql://{db_config['username']}:{password}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ PostgreSQL connection successful!")
            print(f"   Server: {db_config['host']}:{db_config['port']}")
            print(f"   Database: {db_config['database']}")
            print(f"   Schema: {db_config['schema']}")
            print(f"   Version: {version[:50]}...")
            
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("💡 Please check:")
        print("   - PostgreSQL server is running")
        print("   - Database credentials are correct")
        print("   - Database exists")
        print("   - Network connectivity")
        return False
