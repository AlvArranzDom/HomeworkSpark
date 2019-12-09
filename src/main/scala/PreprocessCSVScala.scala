import org.apache.spark.ml.feature.{MinMaxScaler, StringIndexer}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.functions.{col, sum, udf}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType}
import org.apache.spark.sql.{DataFrame, SparkSession}

object PreprocessCSVScala {

  def indexString(df: DataFrame, inputCol: String): DataFrame = {
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(inputCol + "Scaled")

    val indexed_DF = indexer.fit(df)
      .transform(df)
      .drop(inputCol)
      .withColumnRenamed(inputCol + "Scaled", inputCol)

    indexed_DF
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

    spark.sparkContext.setLogLevel("ERROR")

    val file = "1996.csv"
    val inputPath = "src/main/resources/input/"+file

    val csv_DF = spark.read.format("csv").option("header", "true").load(inputPath)

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
      .drop("FlightNum")
      .drop("Year")
      .drop("TailNum")

    // We use this lines to cast the data to the corresponding ones from the dataset
    val casted_DF = cleanInitial_DF.withColumn("Month", col("Month").cast(IntegerType))
      .withColumn("DayofMonth", col("DayofMonth").cast(IntegerType))
      .withColumn("DayOfWeek", col("DayOfWeek").cast(IntegerType))
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

    filtered_DF = filtered_DF.na.drop()
    println("Number of rows after removal: " + filtered_DF.count())

    var indexed_DF = indexString(filtered_DF, "Origin")
    indexed_DF = indexString(indexed_DF, "Dest")
    indexed_DF = indexString(indexed_DF, "UniqueCarrier")

    indexed_DF = minMaxScaler(indexed_DF, "Month")
    indexed_DF = minMaxScaler(indexed_DF, "DayofMonth")
    indexed_DF = minMaxScaler(indexed_DF, "DayOfWeek")
    indexed_DF = minMaxScaler(indexed_DF, "DepTime")
    indexed_DF = minMaxScaler(indexed_DF, "CRSDepTime")
    indexed_DF = minMaxScaler(indexed_DF, "CRSArrTime")
    indexed_DF = minMaxScaler(indexed_DF, "CRSElapsedTime")
    indexed_DF = minMaxScaler(indexed_DF, "ArrDelay")
    indexed_DF = minMaxScaler(indexed_DF, "DepDelay")
    indexed_DF = minMaxScaler(indexed_DF, "Distance")
    indexed_DF = minMaxScaler(indexed_DF, "TaxiOut")

    //println("Printing first 20 rows after indexing.")
    //indexed_DF.show(20)

    //Here we reorder the columns, so the response variables are first and the explanatory is at the end of the DataFrame
    val reordered: Array[String] = Array("Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime", "UniqueCarrier", "CRSElapsedTime", "DepDelay", "Origin", "Dest", "Distance", "TaxiOut", "ArrDelay")
    val df = indexed_DF.select(reordered.head, reordered.tail: _*)

    df.printSchema()
    df.show(20)

    //val outputPath = "src/main/resources/output"
    //df.coalesce(1).write.format("csv").option("header", "true").save(outputPath)
  }
}
