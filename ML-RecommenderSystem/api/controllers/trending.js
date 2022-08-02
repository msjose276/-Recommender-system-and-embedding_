
const Python = require("python-runner");
const fs = require("fs");
const csvParser = require("csv-parser");
const CSVToJSON = require('csvtojson');

pythonScript = "code.py";
dirPath = "./api/pythonCode/";
mlResult = "./api/MLResult/";

const result = [];

const calculateTrending = function(req, res){

    Python.execScript(dirPath + pythonScript, {
            bin: "python",
            //args: ["argument"],
        })
        .then(function (data) {
          fs.createReadStream(mlResult+"result.csv")
                .pipe(csvParser())
                .on("data", (data) => {
                    result.push(data);
                })
                .on("end", () => {
                    console.log(result);
                });
        })
        .catch(function (err) {
          console.log("Error", err);
        });
}

const trending = function(req, res){
    // convert users.csv file to JSON array
    CSVToJSON().fromFile(mlResult+"result.csv")
                    .then(movies => {
                        res.send(movies);
                    }).catch(err => {
                        res.send(err);
                    });
}

module.exports = {
    trending,
    calculateTrending
}