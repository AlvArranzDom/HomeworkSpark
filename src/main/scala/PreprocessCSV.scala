import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{SparkSession, functions}
import org.apache.spark.sql.types.{IntegerType, StringType}

import helpers.DataFrameFunctions.{dropNAValues, normalizedDataFrame}

object PreprocessCSV {
  /**
   * Function that list all the files inside a directory.
   *
   * @param dir directory where to find the csv.
   */
  def getListOfFiles(dir: String): List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  /**
   * Function that rename the csv write output method to a common name.
   *
   * @param dir directory where to find the csv.
   */
  def renameOutputCSV(dir: String): Unit = {
    val listFiles = getListOfFiles(dir)

    for (file <- listFiles) {
      if (file.getName.contains(".csv")) {
        file.renameTo(new File(dir + "/normalized.csv"))
      }
    }
  }

  def main(args: Array[String]) {
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("HomeworkSpark")
      .getOrCreate
    val conf = new SparkConf().setAppName("HomeworkSpark").setMaster("local[*]")

    conf.set("es.index.auto.create", "true")

    val sc = spark.sparkContext
    val sqlc = spark.sqlContext
    spark.sparkContext.setLogLevel("ERROR")

    val inputPath = "src/main/resources/input/"

    val csv_DF = sqlc.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(inputPath + "1996.csv")

    val cleanInitial_DF = csv_DF.drop("ArrTime")
      .drop("ActualElapsedTime")
      .drop("AirTime")
      .drop("TaxiIn")
      .drop("Diverted")
      .drop("CarrierDelay")
      .drop("WeatherDelay")
      .drop("NASDelay")
      .drop("SecurityDelay")
      .drop("LateAircraftDelay")
      .drop("CancellationCode")
      .drop("Year")

    // We use this lines to cast the data to the corresponding ones from the dataset
    val casted_DF = cleanInitial_DF.withColumn("Month", col("Month").cast(IntegerType))
      .withColumn("DayofMonth", col("DayofMonth").cast(IntegerType))
      .withColumn("DayOfWeek", col("DayOfWeek").cast(IntegerType))
      .withColumn("FlightNum", col("FlightNum").cast(StringType))
      .withColumn("TailNum", col("TailNum").cast(StringType))
      .withColumn("DepTime", col("DepTime").cast(IntegerType))
      .withColumn("CRSDepTime", col("CRSDepTime").cast(IntegerType))
      .withColumn("CRSArrTime", col("CRSArrTime").cast(IntegerType))
      .withColumn("CRSElapsedTime", col("CRSElapsedTime").cast(IntegerType))
      .withColumn("ArrDelay", col("ArrDelay").cast(IntegerType))
      .withColumn("Origin", col("Origin").cast(StringType))
      .withColumn("Dest", col("Dest").cast(StringType))
      .withColumn("DepDelay", col("DepDelay").cast(IntegerType))
      .withColumn("Distance", col("Distance").cast(IntegerType))
      .withColumn("UniqueCarrier", col("UniqueCarrier").cast(StringType))
      .withColumn("TaxiOut", col("TaxiOut").cast(IntegerType))
      .withColumn("Cancelled", col("Cancelled").cast(IntegerType))

    // Get non canceled flights.
    var filtered_DF = casted_DF.filter("Cancelled == 0")
    filtered_DF = filtered_DF.drop("Cancelled")

    var indexed_DF = dropNAValues(filtered_DF)

    //Create New Variable for Classification Models Prediction
    indexed_DF = indexed_DF.withColumn("ArrDelayStatus", functions.when(functions.col("ArrDelay") > 0, 1.0).otherwise(0.0))

    // Normalize DataFrame Columns using StringIndexer & MinMaxScales
    val normalized_DF = normalizedDataFrame(indexed_DF, Array("ArrDelay", "ArrDelayStatus"))

    //Here we reorder the columns, so the response variables are first and the explanatory is at the end of the DataFrame
    val reordered: Array[String] = Array("Month", "DayofMonth", "DayOfWeek", "FlightNum", "TailNum", "DepTime", "CRSDepTime", "CRSArrTime",
      "UniqueCarrier", "CRSElapsedTime", "DepDelay", "Origin", "Dest", "Distance", "TaxiOut", "ArrDelay", "ArrDelayStatus")
    val df = normalized_DF.select(reordered.head, reordered.tail: _*)

    df.printSchema()
    df.show(5)

    val outputPath = "src/main/resources/output"
    df.coalesce(1).write.format("csv").option("header", "true").save(outputPath)
    renameOutputCSV(outputPath)

    sqlc.clearCache()
    sc.clearCallSite()
    sc.clearJobGroup()
    spark.close()
  }
}
