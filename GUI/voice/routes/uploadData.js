var express = require('express');
var fs = require("fs");
var router = express.Router();

/* GET users listing. */
router.post('/', function(req, res, next) {
  var data = req.body;
  console.log(data);
});

module.exports = router;
