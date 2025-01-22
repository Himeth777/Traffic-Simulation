// Ensure canvas setup
const canvas = document.getElementById("junctionCanvas");
const ctx = canvas.getContext("2d");

const WIDTH = canvas.width;
const HEIGHT = canvas.height;

const intersectionSize = 80;
const intersectionX = (WIDTH - intersectionSize) / 2;
const intersectionY = (HEIGHT - intersectionSize) / 2;
const SAFE_DISTANCE = 50;

const VEHICLE_LENGTH = 40;
const QUEUE_GAP = 10;

// Vehicle Counters for each lane
const vehicleCounters = {
    horizontal: {
        incoming: 0,
        outgoing: 0
    },
    vertical: {
        incoming: 0,
        outgoing: 0
    }
};

const stopLines = {
    horizontal: {
        incoming: intersectionX - 60,
        outgoing: intersectionX + intersectionSize + 20
    },
    vertical: {
        incoming: intersectionY - 60,
        outgoing: intersectionY + intersectionSize + 20
    }
};

// Lane offsets (fixed positions for lanes)
const laneOffsets = {
    vertical: {
        outgoing: {
            left: WIDTH / 2 - 40,
            right: WIDTH / 2 - 20
        },
        incoming: {
            left: WIDTH / 2,
            right: WIDTH / 2 + 20
        }
    },
    horizontal: {
        incoming: {
            top: HEIGHT / 2 - 40,
            bottom: HEIGHT / 2 - 20
        },
        outgoing: {
            top: HEIGHT / 2,
            bottom: HEIGHT / 2 + 20
        }
    }
};

const spawnPoints = {
    horizontal: {
        incoming: [
            { x: -50, y: laneOffsets.horizontal.incoming.top },
            { x: -50, y: laneOffsets.horizontal.incoming.bottom }
        ],
        outgoing: [
            { x: WIDTH + 50, y: laneOffsets.horizontal.outgoing.top },
            { x: WIDTH + 50, y: laneOffsets.horizontal.outgoing.bottom }
        ]
    },
    vertical: {
        incoming: [
            { x: laneOffsets.vertical.incoming.left, y: -50 },
            { x: laneOffsets.vertical.incoming.right, y: -50 }
        ],
        outgoing: [
            { x: laneOffsets.vertical.outgoing.left, y: HEIGHT + 50 },
            { x: laneOffsets.vertical.outgoing.right, y: HEIGHT + 50 }
        ]
    }
};

const vehicleColors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange'];

// Vehicle class
class Vehicle {
    constructor(x, y, direction, speed, lane, color) {
        this.x = x;
        this.y = y;
        this.direction = direction; // "horizontal" or "vertical"
        this.speed = speed;
        this.lane = lane; // "incoming" or "outgoing"
        this.color = color;
    }

    isColliding(otherVehicle) {
        if (this.direction !== otherVehicle.direction) return false;
        if (this.lane !== otherVehicle.lane) return false;
    
        if (this.direction === "horizontal") {
            return Math.abs(this.x - otherVehicle.x) < SAFE_DISTANCE;
        } else {
            return Math.abs(this.y - otherVehicle.y) < SAFE_DISTANCE;
        }
    }

    getQueuePosition() {
        const vehiclesAhead = vehicles.filter(v => 
            v !== this && 
            v.direction === this.direction && 
            v.lane === this.lane
        ).sort((a, b) => {
            if (this.direction === "horizontal") {
                return this.lane === "incoming" ? b.x - a.x : a.x - b.x;
            } else {
                return this.lane === "incoming" ? b.y - a.y : a.y - b.y;
            }
        });

        return vehiclesAhead.indexOf(this);
    }

    shouldStop() {
        const light = this.getRelevantTrafficLight();
        if (!light) return false;

        const queuePosition = this.getQueuePosition();
        const queueOffset = (VEHICLE_LENGTH + QUEUE_GAP) * queuePosition;

        const vehiclesAhead = vehicles.filter(v => 
            v !== this && 
            v.direction === this.direction && 
            v.lane === this.lane
        ).sort((a, b) => {
            if (this.direction === "horizontal") {
                return this.lane === "incoming" ? a.x - b.x : b.x - a.x;
            } else {
                return this.lane === "incoming" ? a.y - b.y : b.y - a.y;
            }
        });

        if (this.direction === "horizontal") {
            if (this.lane === "incoming") {
                if (light.isRed && this.x <= stopLines.horizontal.incoming - queueOffset) {
                    return true;
                }
                return vehiclesAhead.some(v => 
                    v.x > this.x && 
                    v.x - this.x < VEHICLE_LENGTH + QUEUE_GAP
                );
            } else {
                if (light.isRed && this.x >= stopLines.horizontal.outgoing + queueOffset) {
                    return true;
                }
                return vehiclesAhead.some(v => 
                    v.x < this.x && 
                    this.x - v.x < VEHICLE_LENGTH + QUEUE_GAP
                );
            }
        } else {
            if (this.lane === "incoming") {
                if (light.isRed && this.y <= stopLines.vertical.incoming - queueOffset) {
                    return true;
                }
                return vehiclesAhead.some(v => 
                    v.y > this.y && 
                    v.y - this.y < VEHICLE_LENGTH + QUEUE_GAP
                );
            } else {
                if (light.isRed && this.y >= stopLines.vertical.outgoing + queueOffset) {
                    return true;
                }
                return vehiclesAhead.some(v => 
                    v.y < this.y && 
                    this.y - v.y < VEHICLE_LENGTH + QUEUE_GAP
                );
            }
        }
    }

    getRelevantTrafficLight() {
        if (this.direction === "horizontal") {
            return this.lane === "incoming" ? 
                trafficLights.find(l => l.id === 4) : 
                trafficLights.find(l => l.id === 3);
        } else {
            return this.lane === "incoming" ? 
                trafficLights.find(l => l.id === 2) : 
                trafficLights.find(l => l.id === 1);
        }
    }

    move() {
        if (this.shouldStop()) return;

        if (this.direction === "horizontal") {
            this.x += this.lane === "incoming" ? this.speed : -this.speed;
            if (this.x > WIDTH) this.x = -50;
            if (this.x < -50) this.x = WIDTH;
        } else if (this.direction === "vertical") {
            this.y += this.lane === "incoming" ? this.speed : -this.speed;
            if (this.y > HEIGHT) this.y = -50;
            if (this.y < -50) this.y = HEIGHT;
        }
    }

    draw() {
        ctx.fillStyle = this.color;
        if (this.direction === "vertical") ctx.fillRect(this.x, this.y, 20, 40);
        else if (this.direction === "horizontal") ctx.fillRect(this.x, this.y, 40, 20);
    }
}



// Create vehicles
const vehicles = [
    // Horizontal lane vehicles
    new Vehicle(-50, laneOffsets.horizontal.incoming, "horizontal", 2, "incoming", "red"),
    new Vehicle(WIDTH + 50, laneOffsets.horizontal.outgoing, "horizontal", 3, "outgoing", "blue"),
    // Vertical lane vehicles
    new Vehicle(laneOffsets.vertical.incoming, -50, "vertical", 2, "outgoing", "green"),
    new Vehicle(laneOffsets.vertical.outgoing, HEIGHT + 50, "vertical", 2, "incoming", "yellow"),
];


const trafficLights = [
    { id: 1, x: intersectionX, y: intersectionY + 80, isRed: true, toggleInterval: 2000 },
    { id: 2, x: intersectionX + 40, y: intersectionY - 20, isRed: true, toggleInterval: 2000 },
    { id: 3, x: intersectionX + 80, y: intersectionY + 40, isRed: true, toggleInterval: 2000 },
    { id: 4, x: intersectionX - 20, y: intersectionY, isRed: true, toggleInterval: 2000 }
];


// Draw junction and roads
function drawJunction() {
    

    ctx.fillStyle = "#808080"; // Gray roads

    // Vertical road
    const verticalRoadWidth = 80;
    ctx.fillRect(intersectionX, 0, verticalRoadWidth, HEIGHT);

    // Horizontal road
    const horizontalRoadHeight = 80;
    ctx.fillRect(0, intersectionY, WIDTH, horizontalRoadHeight);

    // Lane dividers
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;

    // Vertical road dividers
    for (let i = 1; i <= 3; i++) {
        const x = intersectionX + i * 20;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, HEIGHT);
        ctx.stroke();
    }

    // Horizontal road dividers
    for (let i = 1; i <= 3; i++) {
        const y = intersectionY + i * 20;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(WIDTH, y);
        ctx.stroke();
    }

    // Centralized intersection box
    ctx.fillStyle = "#555";
    ctx.fillRect(intersectionX, intersectionY, intersectionSize, intersectionSize);

    //traffic light
    
    // Function to draw traffic light rectangles
    function drawTrafficLightRectangles() {
        ctx.fillStyle = "black";
        trafficLights.forEach(light => {
            if (light.id === 1 || light.id === 2) {
                ctx.fillRect(light.x, light.y, 40, 20);
            } else {
                ctx.fillRect(light.x, light.y, 20, 40);
            }
        });
    }

    function drawTrafficLightCircles(x, y, isRed, id) {
        // Red light
        ctx.beginPath();
        ctx.arc(x + 10, y + 5, 5, 0, Math.PI * 2);
        ctx.fillStyle = isRed ? "red" : "gray";
        ctx.fill();
    
        // Green light
        ctx.beginPath();
        if (id === 3 || id === 4) {
            ctx.arc(x + 10, y + 30, 5, 0, Math.PI * 2);
        } else {
            ctx.arc(x + 30, y + 5, 5, 0, Math.PI * 2);
        }
        ctx.fillStyle = isRed ? "gray" : "green";
        ctx.fill();
    }
    
    function drawTrafficLights() {
        drawTrafficLightRectangles();
        trafficLights.forEach(light => {
            drawTrafficLightCircles(light.x, light.y, light.isRed, light.id);
        });
    }

    drawTrafficLights();

}

// Function to display vehicle counters on the canvas
function drawVehicleCounters() {
    ctx.fillStyle = "black";
    ctx.font = "16px Arial";
    let yOffset = 20;

    for (let direction in vehicleCounters) {
        for (let lane in vehicleCounters[direction]) {
            const text = `${direction.charAt(0).toUpperCase() + direction.slice(1)} ${lane.charAt(0).toUpperCase() + lane.slice(1)}: ${vehicleCounters[direction][lane]}`;
            ctx.fillText(text, 10, yOffset);
            yOffset += 20;
        }
    }
}

function updateVehicleCounters() {
    // Reset counters
    for (let direction in vehicleCounters) {
        for (let lane in vehicleCounters[direction]) {
            vehicleCounters[direction][lane] = 0;
        }
    }
    // Count active vehicles
    vehicles.forEach(vehicle => {
        vehicleCounters[vehicle.direction][vehicle.lane]++;
    });
}

// Add activeLight tracker
let activeLight = 1; // Start with light 1 being active (green)

// Modified toggle function to coordinate lights
function toggleTrafficLights() {
    // Set all lights to red first
    trafficLights.forEach(light => {
        light.isRed = true;
    });
    
    // Set current active light to green
    const currentLight = trafficLights.find(light => light.id === activeLight);
    if (currentLight) {
        currentLight.isRed = false;
    }
    
    // Move to next light
    activeLight = (activeLight % 4) + 1;
    
    console.log(`Active light is now ${activeLight}`);
}

// Modified initialization
function initializeTrafficLights() {
    // Set all lights red initially
    trafficLights.forEach(light => light.isRed = true);
    
    // Start the traffic light cycle
    setInterval(toggleTrafficLights, 2000);
}

// Remove individual light intervals, keep animation loop as is
// ...existing code...




function createRandomVehicle() {
    const direction = Math.random() < 0.5 ? "horizontal" : "vertical";
    const lane = Math.random() < 0.5 ? "incoming" : "outgoing";
    const sublane = Math.random() < 0.5 ? 0 : 1; // Choose between two lanes
    const speed = 2;
    const color = vehicleColors[Math.floor(Math.random() * vehicleColors.length)];
    
    const spawn = spawnPoints[direction][lane][sublane];
    return new Vehicle(spawn.x, spawn.y, direction, speed, lane, color);
}

// Initialize vehicle spawn
setInterval(() => {
    vehicles.push(createRandomVehicle());
}, 10000); // 10 seconds

// Reset vehicle counters every 30 seconds
setInterval(() => {
    for (let direction in vehicleCounters) {
        for (let lane in vehicleCounters[direction]) {
            vehicleCounters[direction][lane] = 0;
        }
    }
    console.log('Vehicle counters have been reset.');
}, 30000); // 30000 milliseconds = 30 seconds

// Update animate function
function animate() {
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    drawJunction();
    updateVehicleCounters();
    drawVehicleCounters();
    
    // Update and draw vehicles
    for (let i = vehicles.length - 1; i >= 0; i--) {
        const vehicle = vehicles[i];
        vehicle.move();
        vehicle.draw();
        
        // Optional: Remove vehicles that are too far off screen
        if (vehicle.x < -100 || vehicle.x > WIDTH + 100 || 
            vehicle.y < -100 || vehicle.y > HEIGHT + 100) {

            //vehicleCounters[vehicle.direction][vehicle.lane]++;
            vehicles.splice(i, 1);
        }
    }

    requestAnimationFrame(animate);
}

async function updateTrafficTiming() {
    try {
        const response = await fetch('http://localhost:5000/optimize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ counters: vehicleCounters })
        });
        
        const data = await response.json();
        return data.timing;
    } catch (error) {
        console.error('Error getting ML prediction:', error);
        return 10000; // Default timing
    }
}

// Replace your existing traffic light timer with:
async function adaptiveTrafficLoop() {
    const timing = await updateTrafficTiming();
    toggleTrafficLights();
    setTimeout(adaptiveTrafficLoop, timing);
}

// Start the adaptive control


// Start simulation

animate();
setTimeout(() => {
    adaptiveTrafficLoop();
}, 1000); 
