var loader = '<img src="images/loading.gif" style="height: 50px;">';
var option = '<option value="#VAL">#VAL</option>';
var data = [
  {
    x: ['2013-10-04 22:23:00', '2013-11-04 22:23:00', '2013-12-04 22:23:00'],
    y: [1, 3, 6],
    type: 'scatter'
  }
];

var fileselector;

Plotly.newPlot('emg0', data);
Plotly.newPlot('emg1', data);

$(document).ready(function() {		
	showLoader("spnLoader");
	loadExperiments();
	
});

function loadExperiments(){
	$.ajax({
			type : "GET",
			async : true,
			url : "experiments"
		})
		.done(function(msg) {
			if(msg.status == 1){
				removeLoader("spnLoader");
				var i;
				for(i=0;i<msg.files.length;i++){
					var opt = option;
					opt = opt.replace(/#VAL/g,msg.files[i]);
					$("#source").append(opt);
				}
				loadData(msg.files[0]);
				showToast("Success", "Successfully loaded data.",'#0D638F');
				
			}else{
				removeLoader("spnLoader");
				showToast("Failure", "Unable to load data. Please try after some time.",'#ff2e2e');
				
			}
			
		}).error(function(msg) {
			removeLoader("spnLoader")
			showToast("Error", "Some error occured while loading data. Please try again later.",'#ff2e2e');
		});
}

function showToast(txtheading,txtmessage,colorcode){
	$.toast({
	    heading: txtheading,
	    text: txtmessage,
	    icon: 'info',
	    loader: true,        // Change it to false to disable loader
	    loaderBg: '#9EC600',
	    bgColor: colorcode
	});
}

function showLoader(loaderid){
	$("#"+loaderid).html(loader);
}

function removeLoader(loaderid){
	$("#"+loaderid).html("");
}
//spnLoader

function loadData(filename){
		$.ajax({
			type : "GET",
			async : true,
			url : "experimentdata",
			data:{fname:filename}
		})
		.done(function(msg) {
			if(msg.status == 1){
				removeLoader("spnLoader");
				var i;
				for(i=0;i<msg.files.length;i++){
					var opt = option;
					opt = opt.replace(/#VAL/g,msg.files[i]);
					$("#source").append(opt);
				}
				loadData(msg.files[0]);
				showToast("Success", "Successfully loaded data.",'#0D638F');
				
			}else{
				removeLoader("spnLoader");
				showToast("Failure", "Unable to load data. Please try after some time.",'#ff2e2e');
				
			}
			
		}).error(function(msg) {
			removeLoader("spnLoader")
			showToast("Error", "Some error occured while loading data. Please try again later.",'#ff2e2e');
		});
}

$( "#source" )
  .change(function () {
    var str = "";
    $( "select option:selected" ).each(function() {
      str += $( this ).text() + " ";
    });
    loadData(str);
    console.log(str);
  })
  .change();
