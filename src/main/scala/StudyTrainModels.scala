import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

import helpers.TrainingModelsFunctions.trainModel
import helpers.DataFrameFunctions.createDataFrameForModel

object StudyTrainModels {
  val THRESHOLD = 0.15

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("HomeworkSpark")
      .getOrCreate

    spark.sparkContext.setLogLevel("ERROR")

    val file = "normalized.csv"
    val inputPath = "src/main/resources/output/" + file

    val normalized_DF = spark.read.format("csv").option("header", "true").load(inputPath)

    var df = normalized_DF.withColumn("Month", col("Month").cast(DoubleType))
      .withColumn("DayofMonth", col("DayofMonth").cast(DoubleType))
      .withColumn("DayOfWeek", col("DayOfWeek").cast(DoubleType))
      .withColumn("DepTime", col("DepTime").cast(DoubleType))
      .withColumn("FlightNum", col("FlightNum").cast(DoubleType))
      .withColumn("TailNum", col("TailNum").cast(DoubleType))
      .withColumn("CRSDepTime", col("CRSDepTime").cast(DoubleType))
      .withColumn("CRSArrTime", col("CRSArrTime").cast(DoubleType))
      .withColumn("CRSElapsedTime", col("CRSElapsedTime").cast(DoubleType))
      .withColumn("ArrDelay", col("ArrDelay").cast(DoubleType))
      .withColumn("ArrDelayStatus", col("ArrDelayStatus").cast(DoubleType))
      .withColumn("Origin", col("Origin").cast(DoubleType))
      .withColumn("Dest", col("Dest").cast(DoubleType))
      .withColumn("DepDelay", col("DepDelay").cast(DoubleType))
      .withColumn("Distance", col("Distance").cast(DoubleType))
      .withColumn("UniqueCarrier", col("UniqueCarrier").cast(DoubleType))
      .withColumn("TaxiOut", col("TaxiOut").cast(DoubleType))

    // Train Regression Model
    df = df.drop("ArrDelayStatus")
    val lr_model_DF = createDataFrameForModel(df, "ArrDelay", THRESHOLD)
    trainModel("lr", lr_model_DF)

    // Train Classification Model
    //df = df.drop("ArrDelay")
    //val classification_model_DF = createClassificationModelDataFrame(df, "ArrDelayStatus", THRESHOLD)
    //trainModel("rf", classification_model_DF)
  }
}
