import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{sum, isnan, when, count, col}


object LoadDataScala {

  def main(args: Array[String]) {
    val spark = SparkSession.builder
      .master("local")
      .appName("HomeworkSpark")
      .getOrCreate

    spark.sparkContext.setLogLevel("WARN")

    val path = "src/main/resources/2007.csv"

    val initial_DF = spark.read.format("csv").option("header", "true").load(path)

    val second_DF = initial_DF.drop("ArrTime")
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
      // We use this lines to cast the data to the corresponding ones from the dataset
      .withColumn("Year", col("Year").cast("int")) //Year is the same in the whole DSet...
      .withColumn("Month", col("Month").cast("int"))
      .withColumn("DayofMonth", col("DayofMonth").cast("int"))
      .withColumn("DayOfWeek", col("DayOfWeek").cast("int"))
      .withColumn("DepTime", col("DepTime").cast("int"))
      .withColumn("CRSDepTime", col("CRSDepTime").cast("int"))
      .withColumn("CRSArrTime", col("CRSArrTime").cast("int"))
      .withColumn("FlightNum", col("FlightNum").cast("int"))
      //.withColumn("TailNum", col("TailNum").cast("int"))
      .withColumn("CRSElapsedTime", col("CRSElapsedTime").cast("int"))
      .withColumn("ArrDelay", col("ArrDelay").cast("int"))
      .withColumn("DepDelay", col("DepDelay").cast("int"))
      .withColumn("Distance", col("Distance").cast("int"))
      .withColumn("TaxiOut", col("TaxiOut").cast("int"))
      .withColumn("Cancelled", col("Cancelled").cast("boolean"))
      //this last cast may be unnecesary depending on the ML applied
    // We use this to see NA values.

    second_DF.select(second_DF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()
    println("Number of rows before removal: " + second_DF.count()) // 7453215
    var third_DF = second_DF.na.drop()
    //third_DF.select(second_DF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show() //this is just to check
    println("Number of rows after removal: " + third_DF.count()) // 7275288 are the remaining rows

    //Here we reorder the columns, so the response variables are first and the explanatory is at the end of the DataFrame
    val columns: Array[String] = third_DF.columns //This can be done more efficiently probably...
    val reordered: Array[String] = Array("Year","Month","DayofMonth","DayOfWeek","DepTime","CRSDepTime","CRSArrTime","UniqueCarrier","FlightNum","TailNum","CRSElapsedTime","DepDelay","Origin","Dest","Distance","TaxiOut","Cancelled","ArrDelay")
    var fourth_DF = third_DF.select(reordered.head, reordered.tail: _*)

    //fourth_DF.printSchema()
    //fourth_DF.show(300)
    // TODO: Check Correlations between variables

    // TODO: ML Techniques to predict ArrDelay
  }
}
