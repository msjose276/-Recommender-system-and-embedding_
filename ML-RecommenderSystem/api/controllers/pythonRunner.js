const Python = require("python-runner");



const pythonRunner = function(filePath){

    Python.execScript(filePath, {
        bin: "python",
        args: ["argument"],
      })
        .then(function (data) {
          console.log("data" + data);
          fs.createReadStream("./result.csv")
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
          res.send(err);
        });

        
}