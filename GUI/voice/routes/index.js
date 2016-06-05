var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
	var cookies = req.cookies;
	if(cookies != null){
		if(cookies.user != null){
			if(cookies.user == 1){
				res.render('home');
			}
		}
	}
  	res.render('index');
});

module.exports = router;
