class TrafficOptimizer {
    constructor() {
        this.learningRate = 0.1;
        this.discountFactor = 0.9;
        this.epsilon = 0.1;
        this.qTable = this.loadQTable() || {};
        this.actions = {
            light1: [5000, 10000, 15000],  // Horizontal incoming
            light2: [5000, 10000, 15000],  // Horizontal outgoing
            light3: [5000, 10000, 15000],  // Vertical incoming
            light4: [5000, 10000, 15000]   // Vertical outgoing
        };
    }

    getState(counters) {
        return JSON.stringify({
            horizontalIncoming: counters.horizontal.incoming,
            horizontalOutgoing: counters.horizontal.outgoing,
            verticalIncoming: counters.vertical.incoming,
            verticalOutgoing: counters.vertical.outgoing
        });
    }

    selectAction(state) {
        if (!this.qTable[state]) {
            this.qTable[state] = {
                light1: this.actions.light1.reduce((obj, action) => ({ ...obj, [action]: 0 }), {}),
                light2: this.actions.light2.reduce((obj, action) => ({ ...obj, [action]: 0 }), {}),
                light3: this.actions.light3.reduce((obj, action) => ({ ...obj, [action]: 0 }), {}),
                light4: this.actions.light4.reduce((obj, action) => ({ ...obj, [action]: 0 }), {})
            };
        }

        const timings = {};
        Object.keys(this.actions).forEach(light => {
            if (Math.random() < this.epsilon) {
                timings[light] = this.actions[light][Math.floor(Math.random() * this.actions[light].length)];
            } else {
                const qValues = this.qTable[state][light];
                timings[light] = Object.keys(qValues).reduce((a, b) => qValues[a] > qValues[b] ? a : b);
            }
        });

        return timings;
    }

    update(state, actions, rewards, nextState) {
        if (!this.qTable[nextState]) {
            this.qTable[nextState] = {
                light1: this.actions.light1.reduce((obj, action) => ({ ...obj, [action]: 0 }), {}),
                light2: this.actions.light2.reduce((obj, action) => ({ ...obj, [action]: 0 }), {}),
                light3: this.actions.light3.reduce((obj, action) => ({ ...obj, [action]: 0 }), {}),
                light4: this.actions.light4.reduce((obj, action) => ({ ...obj, [action]: 0 }), {})
            };
        }

        Object.keys(actions).forEach(light => {
            const currentQ = this.qTable[state][light][actions[light]];
            const nextMaxQ = Math.max(...Object.values(this.qTable[nextState][light]));
            this.qTable[state][light][actions[light]] = currentQ + 
                this.learningRate * (rewards[light] + this.discountFactor * nextMaxQ - currentQ);
        });

        this.saveQTable();
    }

    saveQTable() {
        localStorage.setItem('trafficQTable', JSON.stringify(this.qTable));
    }

    loadQTable() {
        const saved = localStorage.getItem('trafficQTable');
        return saved ? JSON.parse(saved) : null;
    }
}