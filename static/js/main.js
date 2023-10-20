// Update the video frame and interact with Plotly.
var socket = io.connect('http://' + document.domain + ':' + location.port);
// Define Plotly plot
var traces = [];
var layout = {title: 'CLIP Similarities'};
Plotly.newPlot('plot_div', traces, layout);

socket.on('frame', function(data) {
  console.log('Received data:', data);

  // Update video frame
  document.getElementById("video_frame").src = "data:image/jpeg;base64," + data.frame;

  // Check if updating frames and lines, or just vertical line because of click
  if (data.hasOwnProperty('set_frame')) {
    // Just click
    console.log("Updating Frame and Line.");
  } else if (data.hasOwnProperty('set_line')) {
    // Just line
    console.log("Updating Line.");
  } else {
    // Get the number of lines from the data
    var num_lines = data.lines.length;

    // If new lines are added, create new traces
    while (traces.length < num_lines) {
      var new_trace = { x: [], y: [], mode: 'lines', name: data.lines[traces.length].title };
      Plotly.addTraces('plot_div', new_trace);
    }
    console.log('Traces:', traces);

    // Update Plotly plot for each line
    for (var i = 0; i < num_lines; i++) {
      Plotly.extendTraces('plot_div', { x: [[data.frame_number]], y: [[data.lines[i].value]]}, [i]);
    }
  }

  // Update vertical line
  var update = {
    'shapes': [
      {
        type: 'line',
        x0: data.frame_number,
        x1: data.frame_number,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: {color: 'red', width: 4}
      }
    ]
  };

  Plotly.relayout('plot_div', update);

  // Emit back the latest frame number
  socket.emit('successful_frame_number', {frame_number: data.frame_number});
});

// Start Button
document.getElementById("start_button").addEventListener("click", function() {
  console.log("Start Button Clicked.");
  socket.emit('start', {data: 'Start'});
});
// Stop Button
document.getElementById("stop_button").addEventListener("click", function() {
  console.log("Stop Button Clicked.");
  socket.emit('stop', {data: 'Stop'});
});

// Clicking on Plot to Jump to Frame
function attachPlotlyClickEvent() {
  var plotDiv = document.getElementById('plot_div');
  plotDiv.on('plotly_click', function(data){
    console.log('Clicked on Plotly plot.');
    var point = data.points[0];
    socket.emit('set_frame', {frame_number: point.x, time: point.x});
  });
}
// Initially attach the event
attachPlotlyClickEvent();

// WebSocket event handling
socket.on('connect', function() {
  console.log('Connected');
});

socket.on('disconnect', function() {
  console.log('Disconnected');
});

socket.on('error', function(error) {
  console.log('Error:', error);
  // Try to connect again
  socket.connect();
  // Alert
  alert("A connection error occurred. Please reload the page.");
  // Emit to server
  socket.emit('client_error', { error: error });
});

// Reset Button
document.getElementById("reset_button").addEventListener("click", function() {
  socket.emit('reset');
  // Reset Plotly plot and any other client-side state
  traces = [];
  layout = {title: 'CLIP Similarities'};
  Plotly.newPlot('plot_div', traces, layout);
  attachPlotlyClickEvent();
});
