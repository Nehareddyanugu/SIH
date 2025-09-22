// models/User.js – defines what a User is

const mongoose = require('mongoose');
const bcrypt = require('bcrypt');

// Schema for User
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },        // User's name
  email: { type: String, required: true, unique: true }, // unique email
  password: { type: String, required: true },     // hashed password
  role: { type: String, default: 'user' }         // could be 'user' or 'admin'
});

// Before saving, hash the password
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) {
    return next();
  }
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

// Method to check password
userSchema.methods.matchPassword = async function(enteredPassword) {
  return await bcrypt.compare(enteredPassword, this.password);
};

module.exports = mongoose.model('User', userSchema);
