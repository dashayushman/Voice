var express = require('express');
var jsonfile = require('jsonfile')
var router = express.Router();
var fs = require('fs');

/* GET users listing. */
router.get('/', function(req, res, next) {
	console.log(req.body.fname);
});

function readFileNames(dirname, onFileContent, onError) {
	
 	return fs.readdirSync(dirname);
}

module.exports = router;
