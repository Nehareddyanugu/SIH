// models/Request.js – what a help request looks like

const mongoose = require('mongoose');

const requestSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true }, // who submitted
  title: { type: String, required: true },        // short title of request
  description: { type: String, required: true },  // detailed description
  category: { type: String },                     // e.g. "Medical", "Food", etc.
  location: { type: String },                     // city or area
  status: { type: String, default: 'Open' },      // e.g. Open, In Progress, Closed
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Request', requestSchema);
