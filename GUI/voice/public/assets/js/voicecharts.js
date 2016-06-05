/* Webarch Admin Dashboard 
/* This JS is Only DEMO Purposes 
-----------------------------------------------------------------*/	
var playing = 0;

$(document).ready(function() {		

	
});

function showToast(txtheading,txtmessage,colorcode){
	$.toast({
	    heading: txtheading,
	    text: txtmessage,
	    icon: 'info',
	    loader: true,        // Change it to false to disable loader
	    loaderBg: colorcode  // To change the background 
	});
}

$( "#btnPlay" ).on( "click", function() {
	if(playing == 0){
		playing = 1;

		showToast("Record", "Started recording data. You can pause data capture anytime by clicking the pause button and resume again by clicking Play.",'#9EC600');
		$( this ).css( "color", "green" );
		$( "#btnPause").css( "color", "" );
	}
  	
});

$( "#btnPause" ).on( "click", function() {
	if(playing == 1){
		playing = 0;
		showToast("Pause", "Paused data capture. You can play again to resume data capture.",'#9EC600');
		$( this ).css( "color", "green" );
		$( "#btnPlay").css( "color", "" );
	}
  	
});

$( "#btnUndo" ).on( "click", function() {
		playing = 0;
		showToast("Undo", "Cleared Data from buffer. You can start recording again by clicking play.",'#9EC600');
		$( "#btnPause").css( "color", "" );
		$( "#btnPlay" ).css( "color", "" );
		clearBufferdata();
});

$( "#btnSave" ).on( "click", function() {
	var expname = $( "#txtExpName").value();
	if(playing == 1){
		showToast("Warning", "Please stop recording before saving your data.",'#ff2e2e');
	}else if(expname == null || expname == ""){
		showToast("Warning", "Please provide an Experiment Name.",'#ff2e2e');
	}else{
		var serObj = new Object();
		serObj.emg = {
			data:emgArr,
			timestamps:emgTimestampArr
		};
		serObj.gyr = {
			data:gyrArr,
			timestamps:gyrTimestampArr
		};
		serObj.ori = {
			data:oriArr,
			timestamps:oriTimestampArr
		};
		serObj.acc = {
			data:accArr,
			timestamps:accTimestampArr
		};
	}
  	
});
