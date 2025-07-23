# Databricks notebook source
# MAGIC %md
# MAGIC ## LDP Pipeline to ingest and transform HL7 data from UC volume

# COMMAND ----------

# MAGIC %pip install hl7apy 

# COMMAND ----------

catalog_name = "danny_catalog"
schema_name = "clinical_coding"
volume_name = "raw_layer"

# COMMAND ----------

# DBTITLE 1,Ingest data with auto loader
import dlt
from pyspark.sql.functions import collect_list, concat_ws, col

@dlt.table
def raw_data():
    raw_df = (spark.readStream
              .format("cloudFiles")
              .option("cloudFiles.format", "text")
              .option("cloudFiles.includeExistingFiles", "true")
              .load(f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/"))
    
    #Group by file path to aggregate each file's content separately
    raw_combined_df = (raw_df
                      .groupBy("_metadata.file_path")
                      .agg(collect_list("value").alias("value"))
                      .withColumn("value", concat_ws("\r", col("value"))))
    
    return raw_combined_df


# COMMAND ----------

from hl7apy.parser import parse_message
from pyspark.sql.functions import udf, col, explode, collect_list, concat_ws
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, MapType

# Define a UDF to parse HL7 messages
def parse_hl7_message(hl7_str):
    # Ensure carriage returns are used (HL7 standard)
    hl7_str = hl7_str.replace('\n', '\r')
    
    try:
        # Parse with find_groups=False to get all segments at the top level
        msg = parse_message(hl7_str, find_groups=False)
        
        # Extract message type and datetime
        message_type = None
        message_datetime = None
        patient_id = None
        patient_name = None
        
        # Extract data from segments
        for seg in msg.children:
            if seg.name == 'MSH':
                try:
                    message_type = seg.msh_9.value
                except:
                    pass
                try:
                    message_datetime = seg.msh_7.value
                except:
                    pass
            elif seg.name == 'PID':
                try:
                    patient_id = seg.pid_3.value
                except:
                    pass
                try:
                    patient_name = seg.pid_5.value
                except:
                    pass
        
        # Extract all segments and their fields
        segments = []
        for seg in msg.children:
            fields = {}
            for field in seg.children:
                # Skip fields that don't have a value
                if not hasattr(field, 'value') or field.value is None:
                    continue
                
                # Add the field to the fields dictionary
                fields[field.name] = field.value
            
            segments.append({"segment_id": seg.name, "fields": fields})
        
    except Exception as e:
        return {
            "message_type": "ERROR",
            "message_datetime": None,
            "patient_id": None,
            "patient_name": None,
            "segments": [{"segment_id": "ERROR", "fields": {"error": str(e)}}]
        }
    
    return {
        "message_type": message_type,
        "message_datetime": message_datetime,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "segments": segments
    }

# Register the UDF
parse_hl7_message_udf = udf(parse_hl7_message, StructType([
    StructField("message_type", StringType(), True),
    StructField("message_datetime", StringType(), True),
    StructField("patient_id", StringType(), True),
    StructField("patient_name", StringType(), True),
    StructField("segments", ArrayType(StructType([
        StructField("segment_id", StringType(), True),
        StructField("fields", MapType(StringType(), StringType()), True)
    ])), True)
]))

# COMMAND ----------

@dlt.table
def bronze():
    return (
        spark.readStream.table("raw_data")
            .withColumn("parsed", parse_hl7_message_udf(col("value")))
    )

# COMMAND ----------

@dlt.table
def silver():
    return (
        spark.readStream.table("bronze").select(
            col("parsed.message_type").alias("message_type"),
            col("parsed.message_datetime").alias("message_datetime"),
            col("parsed.patient_id").alias("patient_id"),
            col("parsed.patient_name").alias("patient_name"),
            col("parsed.segments").alias("segments")
        )
    )

# COMMAND ----------

@dlt.table
def gold_all():
    return (
        spark.readStream.table("silver").select(
            "message_type", 
            "message_datetime",
            "patient_id",
            explode("segments").alias("segment")
        ).select(
            "message_type",
            "message_datetime",
            "patient_id",
            col("segment.segment_id").alias("segment_id"),
            col("segment.fields").alias("fields")
        )
    )

# COMMAND ----------

@dlt.table
def gold_msh():
    return (
        spark.readStream.table("silver").select(
            "message_type", 
            "message_datetime",
            "patient_id",
            explode("segments").alias("segment")
        ).select(
            "message_type",
            "message_datetime",
            "patient_id",
            col("segment.segment_id").alias("segment_id"),
            col("segment.fields").alias("fields")
        ).filter(col("segment_id") == "MSH")
        .select(
            "message_type",
            "message_datetime",
            "patient_id",
            "segment_id",
            "fields.MSH_1",
            "fields.MSH_2",
            "fields.MSH_3",
            "fields.MSH_4",
            "fields.MSH_5",
            "fields.MSH_6",
            "fields.MSH_7",
            "fields.MSH_8",
            "fields.MSH_9",
            "fields.MSH_10",
            "fields.MSH_11",
            "fields.MSH_12",
            "fields.MSH_13",
            "fields.MSH_14",
            "fields.MSH_15",
            "fields.MSH_16",
            "fields.MSH_17",
            "fields.MSH_18",
            "fields.MSH_19",
        )
    )

# COMMAND ----------

@dlt.table
def gold_pid():
    return (
        spark.readStream.table("silver").select(
            "message_type", 
            "message_datetime",
            "patient_id",
            explode("segments").alias("segment")
        ).select(
            "message_type",
            "message_datetime",
            "patient_id",
            col("segment.segment_id").alias("segment_id"),
            col("segment.fields").alias("fields")
        ).filter(col("segment_id") == "PID")
        .select(
            "message_type",
            "message_datetime",
            "patient_id",
            "segment_id",
            "fields.PID_1",
            "fields.PID_2",
            "fields.PID_3",
            "fields.PID_4",
            "fields.PID_5",
            "fields.PID_6",
            "fields.PID_7",
            "fields.PID_8",
            "fields.PID_9",
            "fields.PID_10",
            "fields.PID_11",
            "fields.PID_12",
            "fields.PID_13",
            "fields.PID_14",
            "fields.PID_15",
            "fields.PID_16",
            "fields.PID_17",
            "fields.PID_18",
            "fields.PID_19",
            "fields.PID_20",
            "fields.PID_21",
            "fields.PID_22",
            "fields.PID_23",
            "fields.PID_24",
            "fields.PID_25",
            "fields.PID_26",
            "fields.PID_27",
            "fields.PID_28",
            "fields.PID_29",
            "fields.PID_30"
        )
    )

# COMMAND ----------

@dlt.table
def gold_evn():
    return (
        spark.readStream.table("silver").select(
            "message_type", 
            "message_datetime",
            "patient_id",
            explode("segments").alias("segment")
        ).select(
            "message_type",
            "message_datetime",
            "patient_id",
            col("segment.segment_id").alias("segment_id"),
            col("segment.fields").alias("fields")
        ).filter(col("segment_id") == "EVN")
        .select(
            "message_type",
            "message_datetime",
            "patient_id",
            "segment_id",
            "fields.EVN_1",
            "fields.EVN_2",
            "fields.EVN_3",
            "fields.EVN_4",
            "fields.EVN_5",
            "fields.EVN_6",
            "fields.EVN_7",
        )
    )

# COMMAND ----------

@dlt.table
def gold_pv1():
    return (
        spark.readStream.table("silver").select(
            "message_type", 
            "message_datetime",
            "patient_id",
            explode("segments").alias("segment")
        ).select(
            "message_type",
            "message_datetime",
            "patient_id",
            col("segment.segment_id").alias("segment_id"),
            col("segment.fields").alias("fields")
        ).filter(col("segment_id") == "PV1")
        .select(
            "message_type",
            "message_datetime",
            "patient_id",
            "segment_id",
            "fields.PV1_1",
            "fields.PV1_2",
            "fields.PV1_3",
            "fields.PV1_4",
            "fields.PV1_5",
            "fields.PV1_6",
            "fields.PV1_7",
            "fields.PV1_8",
            "fields.PV1_9",
            "fields.PV1_10",
            "fields.PV1_11",
            "fields.PV1_12",
            "fields.PV1_13",
            "fields.PV1_14",
            "fields.PV1_15",
            "fields.PV1_16",
            "fields.PV1_17",
            "fields.PV1_18",
            "fields.PV1_19",
            "fields.PV1_20",
            "fields.PV1_21",
            "fields.PV1_22",
            "fields.PV1_23",
            "fields.PV1_24",
            "fields.PV1_25",
            "fields.PV1_26",
            "fields.PV1_27",
            "fields.PV1_28",
            "fields.PV1_29",
            "fields.PV1_30",
            "fields.PV1_31",
            "fields.PV1_32",
            "fields.PV1_33",
            "fields.PV1_34",
            "fields.PV1_35",
            "fields.PV1_36",
            "fields.PV1_37",
            "fields.PV1_38",
            "fields.PV1_39",
            "fields.PV1_40",
            "fields.PV1_51",
            "fields.PV1_52",
            "fields.PV1_53",
            "fields.PV1_54"
        )
    )

# COMMAND ----------

@dlt.table
def gold_obr():
    return (
        spark.readStream.table("silver").select(
            "message_type", 
            "message_datetime",
            "patient_id",
            explode("segments").alias("segment")
        ).select(
            "message_type",
            "message_datetime",
            "patient_id",
            col("segment.segment_id").alias("segment_id"),
            col("segment.fields").alias("fields")
        ).filter(col("segment_id") == "OBR")
        .select(
            "message_type",
            "message_datetime",
            "patient_id",
            "segment_id",
            "fields.OBR_1",
            "fields.OBR_2",
            "fields.OBR_3",
            "fields.OBR_4",
            "fields.OBR_5",
            "fields.OBR_6",
            "fields.OBR_7",
            "fields.OBR_8",
            "fields.OBR_9",
            "fields.OBR_10",
            "fields.OBR_11",
            "fields.OBR_12",
            "fields.OBR_13",
            "fields.OBR_14",
            "fields.OBR_15",
            "fields.OBR_16",
            "fields.OBR_17",
            "fields.OBR_18",
            "fields.OBR_19",
            "fields.OBR_20",
            "fields.OBR_21",
            "fields.OBR_22",
            "fields.OBR_23",
            "fields.OBR_24",
            "fields.OBR_25",
            "fields.OBR_26",
            "fields.OBR_27",
            "fields.OBR_28",
            "fields.OBR_29",
            "fields.OBR_30",
            "fields.OBR_31",
            "fields.OBR_32",
            "fields.OBR_33",
            "fields.OBR_34",
            "fields.OBR_35",
            "fields.OBR_36",
            "fields.OBR_37",
            "fields.OBR_38",
            "fields.OBR_39",
            "fields.OBR_40",
            "fields.OBR_41",
            "fields.OBR_42",
            "fields.OBR_43",
            "fields.OBR_44",
            "fields.OBR_45",
            "fields.OBR_46",
            "fields.OBR_47"
        )
    )

# COMMAND ----------

@dlt.table
def gold_obx():
    return (
        spark.readStream.table("silver").select(
            "message_type", 
            "message_datetime",
            "patient_id",
            explode("segments").alias("segment")
        ).select(
            "message_type",
            "message_datetime",
            "patient_id",
            col("segment.segment_id").alias("segment_id"),
            col("segment.fields").alias("fields")
        ).filter(col("segment_id") == "OBX")
        .select(
            "message_type",
            "message_datetime",
            "patient_id",
            "segment_id",
            "fields.OBX_1",
            "fields.OBX_2",
            "fields.OBX_3",
            "fields.OBX_4",
            "fields.OBX_5",
            "fields.OBX_6",
            "fields.OBX_7",
            "fields.OBX_8",
            "fields.OBX_9",
            "fields.OBX_10",
            "fields.OBX_11",
            "fields.OBX_12",
            "fields.OBX_13",
            "fields.OBX_14",
            "fields.OBX_15",
            "fields.OBX_16",
            "fields.OBX_17",
            "fields.OBX_18",
            "fields.OBX_19",
            "fields.OBX_20",
            "fields.OBX_21",
            "fields.OBX_22",
            "fields.OBX_23",
            "fields.OBX_24",
            "fields.OBX_25",
            "fields.OBX_26",
            "fields.OBX_27",
            "fields.OBX_28",
            "fields.OBX_29",
            "fields.OBX_30",
            "fields.OBX_31",
            "fields.OBX_32",
            "fields.OBX_33"
        )
    )

# COMMAND ----------

@dlt.table
def gold_cti():
    return (
        spark.readStream.table("silver").select(
            "message_type", 
            "message_datetime",
            "patient_id",
            explode("segments").alias("segment")
        ).select(
            "message_type",
            "message_datetime",
            "patient_id",
            col("segment.segment_id").alias("segment_id"),
            col("segment.fields").alias("fields")
        ).filter(col("segment_id") == "CTI")
        .select(
            "message_type",
            "message_datetime",
            "patient_id",
            "segment_id",
            "fields.CTI_1",
            "fields.CTI_2",
            "fields.CTI_3",
            "fields.CTI_4"
        )
    )

# COMMAND ----------

@dlt.table
def gold_al1():
    return (
        spark.readStream.table("silver").select(
            "message_type", 
            "message_datetime",
            "patient_id",
            explode("segments").alias("segment")
        ).select(
            "message_type",
            "message_datetime",
            "patient_id",
            col("segment.segment_id").alias("segment_id"),
            col("segment.fields").alias("fields")
        ).filter(col("segment_id") == "AL1")
        .select(
            "message_type",
            "message_datetime",
            "patient_id",
            "segment_id",
            "fields.AL1_1",
            "fields.AL1_2",
            "fields.AL1_3",
            "fields.AL1_4",
            "fields.AL1_5",
            "fields.AL1_6"
        )
    )
