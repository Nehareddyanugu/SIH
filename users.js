// routes/users.js – endpoints for user actions (signup, login, profile)

const express = require('express');
const router = express.Router();
const userController = require('../controllers/userController');
const authMiddleware = require('../middleware/authMiddleware');

// POST /api/users/register — create a new user
router.post('/register', userController.registerUser);

// POST /api/users/login — login, get token
router.post('/login', userController.loginUser);

// GET /api/users/profile — get current user info; protected
router.get('/profile', authMiddleware, userController.getUserProfile);

module.exports = router;
