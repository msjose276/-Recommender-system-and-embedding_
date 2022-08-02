const Python = require("python-runner");
const fs = require("fs");
const csvParser = require("csv-parser");

const pythonScript2 = "code2.py";
const dirPath = "./api/pythonCode/";
const mlResult = "./api/MLResult/";

var result2 = [];
var map = {};

const similarMovies = function(req, res){

    const title = req.body.title;
    process.env["TITLE"] = title;
    console.log("title: "+ title);
    //check if the similarities for the movie title was already computed
    if(map[title]){
        console.log("Already exists in the cach");
        return res.send(map[title]);
    }
    
    Python.execScript(dirPath + pythonScript2, { bin: "python", args: [title]})
                .then(function (data) {
                    result2 = [];
                    //read from the file.
                    fs.createReadStream(mlResult+"result2.csv")
                        .pipe(csvParser())
                        .on("data", (data) => {
                            result2.push(data);
                            map[title] = result2;
                        })
                        .on("end", () => {
                            console.log("result2"+JSON.stringify(map[title]));
                            res.send(map[title]);
                        });
                })
                .catch(function (err) {
                    console.log("Error", err);
                    res.send(err);
                });
}


module.exports = {
    similarMovies
}