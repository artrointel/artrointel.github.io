var eChance10 = 0;
var eCost10 = 0; 
var eChance11 = 0;
var eCost11 = 0;
var eChance12 = 0;
var eCost12 = 0;
var eChance15 = 0;
var eCost15 = 0;
var nSimulation = 0;

var currEnchant = 0;
var currAccumCost = 0;

onmessage = function(e) {
  var startEnchant = e.data[0];
  var endEnchant = e.data[1];
  eChance10 = e.data[2];
  eCost10 = e.data[3];
  eChance11 = e.data[4];
  eCost11 = e.data[5];
  eChance12 = e.data[6];
  eCost12 = e.data[7];
  eChance15 = e.data[8];
  eCost15 = e.data[9];
  nSimulation = e.data[10];
  
  for (var i = 1; i <= nSimulation; i++) {
    currEnchant = startEnchant;
    currAccumCost = 0;
    while (currEnchant != endEnchant) {
      doEnchant(i);
    }
    this.postMessage(["ITERATION", currAccumCost, i]);
  }
  this.postMessage(["FINISHED"]);
}

function doEnchant(iteration) {
    var chance = 0.0;
    var cost = 0;

    if (currEnchant < 10) {
      chance = eChance10;
      cost = eCost10;
    } else if (currEnchant == 10) {
      chance = eChance11;
      cost = eCost11;
    } else if (currEnchant == 11) {
      chance = eChance12;
      cost = eCost12;
    } else {
      chance = eChance15;
      cost = eCost15;
    }
    chance *= 0.01;
  
    if (Math.random() < chance) {
      enchant(true, cost, iteration);
    } else {
      enchant(false, cost, iteration);
    }
}

function enchant(success, cost, iteration) {
  if (success) {
    currEnchant++;
    postMessage(["LOG", "[" + iteration + "회차] " + "+" + currEnchant + " 강화에 성공했습니다."]);
  } else {
    if (0 < currEnchant && currEnchant < 11) {
      currEnchant--;
    } else if (currEnchant >= 11) {
      currEnchant = 10;
    }
    postMessage(["LOG", "[" + iteration + "회차] " + "강화에 실패했습니다."]);
  }
  currAccumCost += cost;
}