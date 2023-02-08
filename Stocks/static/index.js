// $(document).ready(function()
// {
//     const ctx = document.getElementById('myChart_2').getContext('2d');
//     const myChart = new Chart(ctx, {
//         type: 'line',
//         data: {
//             labels: {{ hundredaysavg | safe}},
//             datasets: [{
//                 label: 'Closing Price for {{ name}} from {{ start }} to {{ end }}',
//                 data: {{ hundredaysavg | safe }},
//             }],
          
//         },
//         options:{
//             scales:{
//                 xAxes: [{
//                     display: false //this will remove all the x-axis grid lines
//                 }]
//             }
//         }
//     });
        
// });
// $(document).ready(function()
// {
//     const ctx = document.getElementById('myChart_1').getContext('2d');
//     const myChart = new Chart(ctx, {
//         type: 'line',
//         data: {
//             labels: {{ daysavg | safe}},
//             datasets: [{
//                 label: 'Closing Price for {{ name}} from {{ start }} to {{ end }}',
//                 data: {{ daysavg | safe }},
//             }],
            
//         },
//         options:{
//             scales:{
//                 xAxes: [{
//                     display: false //this will remove all the x-axis grid lines
//                 }]
//             }
//         }
//     });
// });
// $(document).ready(function()
// {
//     const ctx = document.getElementById('myChart').getContext('2d');
//     const myChart = new Chart(ctx, {
//         type: 'line',
//         data: {
//             labels: {{ data | safe}},
//             datasets: [{
//                 label: 'Closing Price for {{ name}} from {{ start }} to {{ end }}',
//                 data: {{ data | safe }}
//             }],
            
//         },
//         options:{
//             scales:{
//                 xAxes: [{
//                     display: false //this will remove all the x-axis grid lines
//                 }]
//             }
//         }
//     });
// });

// var dataFirst = {
//     label: "Actual Price of {{ name }}",
//     data: {{ linear_regression_actual | safe}},
//     borderColor: 'red',
//     lineTension: 0,
//     fill: false
//   };
//   var chartOptions = {
//     legend: {
//       display: true,
//       position: 'top',
//       labels: {
//         boxWidth: 80,
//         fontColor: 'black'
//       }
//     }
//   };
// var dataSecond = {
//     label: "Predicted Price of {{ name }}",
//     data: {{ linear_regression_predicted | safe }},
//     borderColor: 'blue'
//   };
//   var speedData = {
//     labels: [dataFirst, dataSecond],
//     datasets: [dataFirst, dataSecond]
//   };
//   $(document).ready(function()
//     {
//     const ctx = document.getElementById('myChart_3').getContext('2d');
//     const myChart = new Chart(ctx, {
//         type: 'line',
//         data: speedData,
//         borderColor: 'red',
//         lineTension: 0,
//         fill: false
//         },
//         options:{
//             scales:{
//                 xAxes: [{
//                     display: false //this will remove all the x-axis grid lines
//                 }]
//             }
//         }
//     });