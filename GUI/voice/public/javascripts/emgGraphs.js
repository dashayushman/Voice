Myo.on('connected', function(){
	console.log('connected');
	this.streamEMG(true);

	setInterval(function(){
		updateEmgGraph(rawEmgData);
	}, 25)

	setInterval(function(){
		updateGyrGraph(rawGyroData);
	}, 25)

	setInterval(function(){
		updateAccGraph(rawAccData);
	}, 25)
})

Myo.connect('com.myojs.emgGraphs');

var SerielizedBufferData;

var arm = '';

var emgArr = [];
var emgTimestampArr = [];

var gyrArr = [];
var gyrTimestampArr = [];

var accArr = [];
var accTimestampArr = [];

var oriArr = [];
var oriTimestampArr = [];

var rawEmgData = [0,0,0,0,0,0,0,0];
var rawGyroData = [0,0,0];
var rawAccData = [0,0,0];
var rawOriData = [0,0,0,0];

Myo.on('emg', function(data,timestamp){
	rawEmgData = data;
	if(playing == 1){
		emgArr.push(data);
		emgTimestampArr.push(timestamp);
	}
});

Myo.on('gyroscope', function(quant,timestamp){
	rawGyroData[0] = quant.x;
	rawGyroData[1] = quant.y;
	rawGyroData[2] = quant.z;

	if(playing == 1){
		gyrArr.push(rawGyroData);
		gyrTimestampArr.push(timestamp);
	}
});


Myo.on('accelerometer', function(data,timestamp){
   	rawAccData[0] = data.x;
	rawAccData[1] = data.y;
	rawAccData[2] = data.z;

	if(playing == 1){
		accArr.push(rawAccData);
		accTimestampArr.push(timestamp);
	}
});

Myo.on('orientation', function(data,timestamp){
   	rawOriData[0] = data.x;
	rawOriData[1] = data.y;
	rawOriData[2] = data.z;
	rawOriData[3] = data.w;

	if(playing == 1){
		oriArr.push(rawOriData);
		oriTimestampArr.push(timestamp);
	}
});


var emgrange = 150;
var emgresolution = 50;
var emgGraphs;

var gyrrange = 500;
var gyrresolution = 100;
var gyrGraphs;

var accrange = 5;
var accresolution = 100;
var accGraphs;

var emggraphData= [
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0)
];

var gyrgraphData= [
	Array.apply(null, Array(gyrresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(gyrresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(gyrresolution)).map(Number.prototype.valueOf,0)
];

var accgraphData= [
	Array.apply(null, Array(accresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(accresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(accresolution)).map(Number.prototype.valueOf,0)
]

$(document).ready(function(){

	emgGraphs = emggraphData.map(function(podData, podIndex){
		return $('#pod' + podIndex).plot(formatEmgFlotData(podData), {
			colors: ['#8aceb5'],
			xaxis: {
				show: false,
				min : 0,
				max : emgresolution
			},
			yaxis : {
				min : -emgrange,
				max : emgrange,
			},
			grid : {
				borderColor : "#427F78",
				borderWidth : 1
			}
		}).data("plot");
	});

	gyrGraphs = gyrgraphData.map(function(gyrData, gyrIndex){
		return $('#gyr' + gyrIndex).plot(formatGyrFlotData(gyrData), {
		colors: [ '#04fbec'],
		xaxis: {
			show: false,
			min : 0,
			max : gyrresolution
		},
		yaxis : {
			min : -gyrrange,
			max : gyrrange,
		},
		grid : {
			borderColor : "#427F78",
			borderWidth : 1
		}
	}).data("plot");
		});

	accGraphs = accgraphData.map(function(accData, accIndex){
		return $('#acc' + accIndex).plot(formatAccFlotData(accData), {
		colors: [ '#04fbbc'],
		xaxis: {
			show: false,
			min : 0,
			max : accresolution
		},
		yaxis : {
			min : -accrange,
			max : accrange,
		},
		grid : {
			borderColor : "#427F78",
			borderWidth : 1
		}
		}).data("plot");
	});

});

var formatEmgFlotData = function(data){
		return [data.map(function(val, index){
				return [index, val]
			})];
}


var formatGyrFlotData = function(data){
	return [data.map(function(val, index){
				return [index, val]
			})];
}

var formatAccFlotData = function(data){
	return [data.map(function(val, index){
				return [index, val]
			})];
}



var updateEmgGraph = function(emgData){

	emggraphData.map(function(data, index){
		emggraphData[index] = emggraphData[index].slice(1);
		emggraphData[index].push(emgData[index]);
		emgGraphs[index].setData(formatEmgFlotData(emggraphData[index]));
		emgGraphs[index].draw();
	});

}

var updateGyrGraph = function(orientationData){
	gyrgraphData.map(function(data, index){
		gyrgraphData[index] = gyrgraphData[index].slice(1);
		gyrgraphData[index].push(orientationData[index]);
		gyrGraphs[index].setData(formatGyrFlotData(gyrgraphData[index]));
		gyrGraphs[index].draw();
	});

}

var updateAccGraph = function(accData){
	accgraphData.map(function(data, index){
		accgraphData[index] = accgraphData[index].slice(1);
		accgraphData[index].push(accData[index]);
		accGraphs[index].setData(formatAccFlotData(accgraphData[index]));
		accGraphs[index].draw();
	});
}

function clearBufferdata(){
	 emgArr = [];
	 emgTimestampArr = [];

	 gyrArr = [];
	 gyrTimestampArr = [];

	 accArr = [];
	 accTimestampArr = [];

	 oriArr = [];
	 oriTimestampArr = [];
}

function recordArm(){
	arm = Myo.arm;
}

function serielizeBufferData(expname){
	SerielizedBufferData = new Object();
		SerielizedBufferData.arm = arm;
		SerielizedBufferData.expName= expname;
		SerielizedBufferData.emg = {
			data:emgArr,
			timestamps:emgTimestampArr
		};
		SerielizedBufferData.gyr = {
			data:gyrArr,
			timestamps:gyrTimestampArr
		};
		SerielizedBufferData.ori = {
			data:oriArr,
			timestamps:oriTimestampArr
		};
		SerielizedBufferData.acc = {
			data:accArr,
			timestamps:accTimestampArr
		};
}

function uploadBufferData(){
	$.ajax({
			type : "POST",
			async : true,
			url : "uploadData",
			data: JSON.stringify(SerielizedBufferData)
		})
		.done(function(msg) {
			if(msg.status == 1){
				clearBufferdata();
				$( "#txtExpName").val("");
				removeLoader("spnLoader")
				showToast("Success", "Successfully uploaded data. You can start a new experiment or start annotating or visualising your data.",'#0D638F');
				$( "#btnPlay").css( "color", "" );
				$( "#btnPause").css( "color", "" );
			}else{
				clearBufferdata();
				$( "#txtExpName").val("");
				removeLoader("spnLoader")
				showToast("Success", "Unable to upload data. Please try after some time.",'#ff2e2e');
				$( "#btnPlay").css( "color", "" );
				$( "#btnPause").css( "color", "" );
			}
			
		}).error(function(msg) {
			clearBufferdata();
			$( "#txtExpName").val("");
			removeLoader("spnLoader")
			showToast("Error", "Some error occured while uploading data. Please try again later.",'#ff2e2e');
			$( "#btnPlay").css( "color", "" );
			$( "#btnPause").css( "color", "" );
		});
}