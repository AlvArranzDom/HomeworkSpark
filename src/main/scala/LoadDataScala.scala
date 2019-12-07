import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{sum, isnan, when, count, col}


object LoadDataScala {

  def main(args: Array[String]) {
    val spark = SparkSession.builder
      .master("local")
      .appName("HomeworkSpark")
      .getOrCreate

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
    // We use this to see NA values.

    // TODO: Look for missing values.
    second_DF.select(second_DF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()
    printf("Number of rows before removal: " + second_DF.count()) // 7453215
    var third_DF = second_DF.na.drop();
    // third_DF.select(second_DF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show() //this is unnecesary
    printf("Number of rows after removal: " + third_DF.count()) // 7453193

    // We can see that there are only 22 missing values in TailNum and 0 in the rest of the variables

    // TODO: Check Correlations between variables
    //initial_DF.printSchema()
    //third_DF.show(100) //This shows you the dataframe in a table like format an rows u ask it
   }
}
