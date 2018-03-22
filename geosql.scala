import com.vividsolutions.jts.geom.{Coordinate, Geometry, GeometryFactory}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.datasyslab.geospark.formatMapper.shapefileParser.ShapefileReader
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator
import org.datasyslab.geospark.spatialRDD.SpatialRDD
import org.datasyslab.geospark.utils.GeoSparkConf
import org.datasyslab.geosparksql.utils.{Adapter, GeoSparkSQLRegistrator}




  var sparkSession:SparkSession = SparkSession.builder().config("spark.serializer",classOf[KryoSerializer].getName).
    config("spark.kryo.registrator", classOf[GeoSparkKryoRegistrator].getName).
    master("local[*]").appName("GeoSparkSQL-demo").getOrCreate()
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  GeoSparkSQLRegistrator.registerAll(sparkSession.sqlContext)


  val resourceFolder = "/Users/htogashi/IdeaProjects/GeoSparkTemplateProject/geospark-sql/scala/src/test/resources/"

  val csvPolygonInputLocation = resourceFolder + "testenvelope.csv"
  val csvPointInputLocation = resourceFolder + "testpoint.csv"
  val shapefileInputLocation = resourceFolder + "shapefiles/dbf"


    val geosparkConf = new GeoSparkConf(sparkSession.sparkContext.getConf)
    println(geosparkConf)

////////////////////////

var spatialRDD = new SpatialRDD[Geometry]
spatialRDD.rawSpatialRDD = ShapefileReader.readToGeometryRDD(sparkSession.sparkContext, resourceFolder + "shapefiles/dbf")
var mapPol = Adapter.toDf(spatialRDD,sparkSession)
  mapPol.createOrReplaceTempView("polydf0")
var polydf = sparkSession.sql("select ST_GeomFromWKT(rddshape) as polyshape from polydf0")
polydf.createOrReplaceTempView("polydf1")
polydf.show()

var spatialRDD2 = new SpatialRDD[Geometry]
spatialRDD2.rawSpatialRDD = ShapefileReader.readToGeometryRDD(sparkSession.sparkContext, resourceFolder + "shapefiles/point")
var mapPoint = Adapter.toDf(spatialRDD2,sparkSession)
mapPoint.createOrReplaceTempView("pointdf0")
var pointdf = sparkSession.sql("select ST_GeomFromWKT(rddshape) as pointshape from pointdf0")
pointdf.createOrReplaceTempView("pointdf1")
pointdf.show()

var rangeJoinDf = sparkSession.sql("select * from polydf1, pointdf1 where ST_Contains(polydf1.polyshape, pointdf1.pointshape) ")

rangeJoinDf.select().count()
rangeJoinDf.printSchema()

  def testDistanceJoinQuery(): Unit =
  {
    val geosparkConf = new GeoSparkConf(sparkSession.sparkContext.getConf)
    println(geosparkConf)

    var pointCsvDF1 = sparkSession.read.format("csv").option("delimiter",",").option("header","false").load(csvPointInputLocation)
    pointCsvDF1.createOrReplaceTempView("pointtable")
    pointCsvDF1.show()
    var pointDf1 = sparkSession.sql("select ST_Point(cast(pointtable._c0 as Decimal(24,20)),cast(pointtable._c1 as Decimal(24,20)), \"myPointId\") as pointshape1 from pointtable")
    pointDf1.createOrReplaceTempView("pointdf1")
    pointDf1.show()

    var pointCsvDF2 = sparkSession.read.format("csv").option("delimiter",",").option("header","false").load(csvPointInputLocation)
    pointCsvDF2.createOrReplaceTempView("pointtable")
    pointCsvDF2.show()
    var pointDf2 = sparkSession.sql("select ST_Point(cast(pointtable._c0 as Decimal(24,20)),cast(pointtable._c1 as Decimal(24,20)), \"myPointId\") as pointshape2 from pointtable")
    pointDf2.createOrReplaceTempView("pointdf2")
    pointDf2.show()

    var distanceJoinDf = sparkSession.sql("select * from pointdf1, pointdf2 where ST_Distance(pointdf1.pointshape1,pointdf2.pointshape2) < 2")
    distanceJoinDf.explain()
    distanceJoinDf.show(10)
    assert (distanceJoinDf.count()==2998)
  }

  def testAggregateFunction(): Unit =
  {
    val geosparkConf = new GeoSparkConf(sparkSession.sparkContext.getConf)
    println(geosparkConf)

    var pointCsvDF = sparkSession.read.format("csv").option("delimiter",",").option("header","false").load(csvPointInputLocation)
    pointCsvDF.createOrReplaceTempView("pointtable")
    var pointDf = sparkSession.sql("select ST_Point(cast(pointtable._c0 as Decimal(24,20)), cast(pointtable._c1 as Decimal(24,20)), \"myPointId\") as arealandmark from pointtable")
    pointDf.createOrReplaceTempView("pointdf")
    var boundary = sparkSession.sql("select ST_Envelope_Aggr(pointdf.arealandmark) from pointdf")
    val coordinates:Array[Coordinate] = new Array[Coordinate](5)
    coordinates(0) = new Coordinate(1.1,101.1)
    coordinates(1) = new Coordinate(1.1,1100.1)
    coordinates(2) = new Coordinate(1000.1,1100.1)
    coordinates(3) = new Coordinate(1000.1,101.1)
    coordinates(4) = coordinates(0)
    val geometryFactory = new GeometryFactory()
    geometryFactory.createPolygon(coordinates)
    assert(boundary.take(1)(0).get(0)==geometryFactory.createPolygon(coordinates))
  }


