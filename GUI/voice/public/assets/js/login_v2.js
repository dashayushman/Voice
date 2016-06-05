/* Webarch Admin Dashboard 
-----------------------------------------------------------------*/ 
var loader = '<img src="images/loading.gif" style="height: 60px;margin-top: 0px;margin-bottom: 8px;">';
var errmsg = '<span id="iderror" style="color: #COL;">#ERRMSG</span>';
var errbordercol = 'border-bottom-color: red;';
$(document).ready(function() {		
	$('#frm_login').submit(function( event ){
        removeErrmsg("iderror");
        stopLoader("idloader");
        startLoader("idloader");
        var username = $( "#login_username" ).val();
        var password = $( "#login_pass" ).val();
		

        if(username =="" || username == null){
            event.preventDefault();
            removeErrmsg("iderror");
            stopLoader("idloader");
            showErrorMsg("iderrmsgdiv","Please enter a username to login","red");
            underlineElement("login_username","red");
            return;
        }else{
            removeErrmsg("iderror");
            stopLoader("idloader");
            removeUnderlineElement("login_username");
            
        }

        if(password =="" || password == null){
            event.preventDefault();
            removeErrmsg("iderror");
            stopLoader("idloader");
            showErrorMsg("iderrmsgdiv","Please enter a password to login","red");
            underlineElement("login_pass","red");
            return;
        }else{
            removeErrmsg("iderror");
            stopLoader("idloader");
            removeUnderlineElement("login_pass")
        }
        //event.preventDefault();
        startLoader("idloader");
        $.ajax({
            type : "POST",
            async : false,
            url : "/validateLogin",
            data: "username=" + username + "&password=" + password
        })
        .done(function(msg) {
            if(msg.status == 1){
                removeErrmsg("iderror");
                removeUnderlineElement("login_username");
                removeUnderlineElement("login_pass");
            }else{
                event.preventDefault();
                removeErrmsg("iderror");
                stopLoader("idloader");
                showErrorMsg("iderrmsgdiv","Please enter a valid username or password to login","red");
            }
        }).error(function(msg) {
            event.preventDefault();
            removeErrmsg("iderror");
            stopLoader("idloader");
            showErrorMsg("iderrmsgdiv","Some error occured while validating","red");
        });
        
        //make ajax call
        //validate and redirect to next page
	});

    $("#login_username").on("change paste keyup", function(e) {
       if(this.value =="" || this.value == null){
            removeErrmsg("iderror");
            stopLoader("idloader");
            showErrorMsg("iderrmsgdiv","Please enter a username to login","red");
            underlineElement("login_username","red");
            return;
        }else{
            removeErrmsg("iderror");
            stopLoader("idloader");
            removeUnderlineElement("login_username");
            return;
        }
    });
    $("#login_pass").on("change paste keyup", function(e) {
        
       if(this.value =="" || this.value == null){
            removeErrmsg("iderror");
            stopLoader("idloader");
            showErrorMsg("iderrmsgdiv","Please enter a password to login","red");
            underlineElement("login_pass","red");
            return;
        }else{
            removeErrmsg("iderror");
            stopLoader("idloader");
            removeUnderlineElement("login_pass")
            return;
        }
    });
    });

function startLoader(loaderid){
    $( "#"+loaderid ).html(loader);
}

function stopLoader(loaderid){
    $( "#"+loaderid ).html("");
}

function underlineElement(elid,colour){
    $( "#"+elid ).css("border-bottom-color",colour);
}

function removeUnderlineElement(elid){
    $( "#"+elid ).css("border-bottom-color","");
}

function showErrorMsg(errid,err,col){
    errortag = errmsg;
    errortag = errortag.replace(/#COL/g,col);
    errortag = errortag.replace(/#ERRMSG/g,err);
    $( "#"+errid ).append(errortag);
}
function removeErrmsg(errid){
     $( "#"+errid ).remove();
}