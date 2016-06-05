//This tells Myo.js to create the web sockets needed to communnicate with Myo Connect


Myo.on('connected', function(){
	console.log('connected');
	this.streamEMG(true);

	setInterval(function(){
		updateEmgGraph(rawEmgData);
	}, 25)

	setInterval(function(){
		updateGyroGraph(rawEmgData);
	}, 25)
})

Myo.connect('com.myojs.emgGraphs');


var rawEmgData = [0,0,0,0,0,0,0,0];
var rawGyroData = [0,0,0,0];

Myo.on('emg', function(data){
	rawEmgData = data;
})

Myo.on('gyroscope', function(quant){
	rawGyroData = quant;
	//updateGyroGraph(quant);
})

var emgrange = 150;
var emgresolution = 50;
var emgGraphs;

var graphData= [
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(emgresolution)).map(Number.prototype.valueOf,0)
]

$(document).ready(function(){

	emgGraphs = graphData.map(function(podData, podIndex){
		return $('#pod' + podIndex).plot(formatFlotData(podData), {
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


});

var formatFlotData = function(data){
		return [data.map(function(val, index){
				return [index, val]
			})]
}


var updateEmgGraph = function(emgData){

	graphData.map(function(data, index){
		graphData[index] = graphData[index].slice(1);
		graphData[index].push(emgData[index]);

		emgGraphs[index].setData(formatFlotData(graphData[index]));
		emgGraphs[index].draw();


	})

}




/*




*/