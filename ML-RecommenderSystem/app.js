require("dotenv").config();
var express = require("express");
const fs = require("fs");
const Router = require('./api/routes/index');
const trending = require('./api/controllers/trending');
var app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(function(req, res,next){
  console.log('method= '+ req.method+ ' - url = '+req.url)
  next();
})



const trendingResultFile = './api/MLResult/result.csv'

fs.access(trendingResultFile, fs.F_OK, (err) => {
  if (err) {
    console.log("The trending file does not exist. Creating one....")
    trending.calculateTrending();
    return
  }
  console.log("The trending file exist")
})


app.use("/api",function(req, res, next){
  res.header(process.env.CORS_ACAO, process.env.ORIGIN_ALLOWED_ON_CORS);
  res.header(process.env.CORS_ACAH, process.env.HEADERS_ALLOWED_ON_CORS);
  res.header(process.env.CORS_ACAM, process.env.METHODS_ALLOWED_ON_CORS);
  next();
});

app.use('/api',Router);

const server = app.listen(process.env.PORT, function (req, res) {
  console.log("Listening to port: " + server.address().port);
});
