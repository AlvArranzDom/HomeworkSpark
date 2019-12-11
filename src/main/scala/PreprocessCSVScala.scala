import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{MinMaxScaler, StringIndexer}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.functions.{col, sum, udf}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType}
import org.apache.spark.sql.{DataFrame, SparkSession, functions}

object PreprocessCSVScala {

  def indexString(df: DataFrame, inputCol: String): DataFrame = {
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(inputCol + "Scaled")

    val indexed_DF = indexer.fit(df)
      .transform(df)
      .drop(inputCol)
      .withColumnRenamed(inputCol + "Scaled", inputCol)

    minMaxScaler(indexed_DF, inputCol)
  }

  def minMaxScaler(df: DataFrame, inputCol: String): DataFrame = {
    val vectorizeCol = udf((v: Double) => Vectors.dense(Array(v)))
    val headValue = udf((arr: DenseVector) => arr.toArray(0))
    val newDF = df.withColumn(inputCol + "Vec", vectorizeCol(df(inputCol)))

    val scalerArrDelay = new MinMaxScaler()
      .setInputCol(inputCol + "Vec")
      .setOutputCol(inputCol + "Scaled")
      .setMax(1)
      .setMin(0)

    val scaled_DF = scalerArrDelay.fit(newDF)
      .transform(newDF)
      .drop(inputCol)
      .drop(inputCol + "Vec")
      .withColumnRenamed(inputCol + "Scaled", inputCol)

    scaled_DF.withColumn(inputCol, headValue(col(inputCol)).cast(DoubleType))
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
      .load(inputPath + "2007.csv")

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

    // We use this to see NA values.
    filtered_DF.select(filtered_DF.columns.map(c => sum(col(c).isNull.cast(IntegerType)).alias(c)): _*).show()
    println("Number of rows before removal: " + filtered_DF.count())

    var indexed_DF = filtered_DF.na.drop()
    println("Number of rows after removal: " + indexed_DF.count())

    //Comment this line to generate a Dataset for Linear Regression
    indexed_DF = indexed_DF.withColumn("ArrDelay", functions.when(functions.col("ArrDelay") > 0, 1.0).otherwise(0.0))

    val colNamesList = indexed_DF.columns
    val columnDataTypes: Array[String] = indexed_DF.schema.fields.map(x => x.dataType).map(x => x.toString)

    for (i <- colNamesList.indices) {
      if (colNamesList(i) != "ArrDelay") {
        if (columnDataTypes(i) == "StringType") {
          indexed_DF = indexString(indexed_DF, colNamesList(i))
        } else {
          indexed_DF = minMaxScaler(indexed_DF, colNamesList(i))
        }
      }
    }

    //Here we reorder the columns, so the response variables are first and the explanatory is at the end of the DataFrame
    val reordered: Array[String] = Array("Month", "DayofMonth", "DayOfWeek", "FlightNum", "TailNum", "DepTime", "CRSDepTime", "CRSArrTime",
      "UniqueCarrier", "CRSElapsedTime", "DepDelay", "Origin", "Dest", "Distance", "TaxiOut", "ArrDelay")
    val df = indexed_DF.select(reordered.head, reordered.tail: _*)

    df.printSchema()
    df.show(5)

    val outputPath = "src/main/resources/outputLinearRegression"
    df.coalesce(1).write.format("csv").option("header", "true").save(outputPath)

    sqlc.clearCache()
    sc.clearCallSite()
    sc.clearJobGroup()
    spark.close()
  }
}
