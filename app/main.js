const express = require('express');
const app = express();
const port = 80;

var fs = require('fs');


const bodyParser = require('body-parser');


app.engine('.ejs', require('ejs').__express)
app.set('views', __dirname + '/index')
app.use(express.static(__dirname + '/index'))
app.set('view engine', 'ejs')

app.use(bodyParser.json());
app.get('/minimax', function(req, res, next) {
    res.render('minimax-alpha-beta.ejs')
})
app.get('/random', function(req, res, next) {
  res.render('random.ejs')
})
app.get('/', function(req, res, next) {
    res.render('reinforced.ejs')
  })
app.post("/send", (req, res) => {

  var obj={};
  console.log('body: ' + JSON.stringify(req.body));
  fs.writeFile("games.json", JSON.stringify(req.body), function(err) {
    if (err) {
        console.log(err);
    }
    else{console.log("Saved");}
});
});
app.listen(port, () => console.log('Server'))