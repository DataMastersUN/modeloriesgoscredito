const chartData = { 
    labels: ["Estado de préstamo","XIngresos","Grado","Z"], 
    data: [50,30,10,10] };
const myChart = document.querySelector(".my-chart");
const ul = document.querySelector(".programming-stats .details ul");



new Chart(myChart,{
    type: "doughnut",
    data: {
        labels: chartData.labels,
        datasets: [{
            labels: "Pruebas de graficos",
            data: chartData.data,
        }]
    },
    options: {
        borderWidth: 10,
        borderRadius: 2,
        hoverBorderWidth: 0,
        plugins:{
            legend: {
                display: false,
            }
        }
    }
});

const populateUl = () => {
    chartData.labels.forEach((l, i) => {
        let li = document.createElement("li");
        li.innerHTML = `${l}: <span class='percentage'>${chartData.data[i]}%</span>`;
        ul.appendChild(li);
    }); 
};

populateUl();
