import React, { useRef, useEffect, useState, useCallback } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-converter';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { drawKeypoints, drawSkeleton } from './utilities';
import { triggerAIFeedback } from './aiSocialAgent';
import mockUserData from './mockUserData';

function App() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [loading, setLoading] = useState(true);

    // Exercise State Management
    const [currentExerciseTracker, setCurrentExerciseTracker] = useState('Right Arm Raise');
    const [reps, setReps] = useState(0);
    const [instruction, setInstruction] = useState('Raise right arm to shoulder height');

    // AI Social Agent state
    const [aiMessage, setAiMessage] = useState('');
    const aiMessageTimer = useRef(null);

    // Show AI message and set a fallback auto-hide timeout (20 seconds), usually hidden by aiSocialAgent after playback finishes
    const showAiMessage = useCallback((text) => {
        setAiMessage(text);
        if (aiMessageTimer.current) clearTimeout(aiMessageTimer.current);
        // Force hide after 20 seconds by default to prevent the box from staying on screen if the voice engine freezes
        aiMessageTimer.current = setTimeout(() => setAiMessage(''), 20000);
    }, []);

    // Actively hide AI message
    const hideAiMessage = useCallback(() => {
        setAiMessage('');
        if (aiMessageTimer.current) clearTimeout(aiMessageTimer.current);
    }, []);

    // Use useRef for values that need to be accessed inside the requestAnimationFrame loop
    // without causing constant re-renders or stale closures
    const exerciseState = useRef({
        currentExerciseType: 'right_arm', // 'right_arm', 'left_arm', 'both_arms', 'side_both_arms', 'side_bends', 'neck_side_bends'
        reps: 0,
        status: 'resting', // 'resting', 'holding', 'completed'
        holdStartTime: 0,
        targetHoldTime: 3000, // 3 seconds
        graceFrames: 0,
        lastSideBendDirection: null, // Track alternating side bends ('left' or 'right')
    });

    useEffect(() => {
        const runPoseDetection = async () => {
            try {
                await tf.setBackend('webgl');
                // Ensure backend is ready
                await tf.ready();
                console.log('TensorFlow JS Backend Ready');

                const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING };
                const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
                console.log('Pose Detector Loaded');

                setLoading(false);

                const setupCamera = async () => {
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        throw new Error('Browser API navigator.mediaDevices.getUserMedia not available');
                    }

                    const video = videoRef.current;
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            width: 360,
                            height: 640
                        },
                        audio: false,
                    });

                    video.srcObject = stream;

                    return new Promise((resolve) => {
                        video.onloadedmetadata = () => {
                            video.play();
                            resolve(video);
                        };
                    });
                };

                await setupCamera();

                const detect = async () => {
                    const video = videoRef.current;
                    const canvas = canvasRef.current;

                    if (video && video.readyState === 4 && canvas) {
                        const videoWidth = video.videoWidth;
                        const videoHeight = video.videoHeight;

                        // Force set internal dimensions to match video stream
                        video.width = videoWidth;
                        video.height = videoHeight;
                        canvas.width = videoWidth;
                        canvas.height = videoHeight;

                        try {
                            const poses = await detector.estimatePoses(video);

                            const ctx = canvas.getContext('2d');
                            ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas

                            if (poses.length > 0) {
                                // Check if any pose has a score (sometimes it's global, sometimes per keypoint)
                                // MoveNet usually returns one pose.
                                console.log('Poses found:', poses); // Added logging for poses
                                // Draw the skeleton and keypoints
                                poses.forEach(pose => {
                                    // Lower threshold to 0.1 for debugging
                                    drawSkeleton(pose.keypoints, 0.1, ctx);
                                    drawKeypoints(pose.keypoints, 0.1, ctx);

                                    // Extract keypoints needed for Arm Raise exercise
                                    const leftShoulder = pose.keypoints.find(k => k.name === 'left_shoulder');
                                    const rightShoulder = pose.keypoints.find(k => k.name === 'right_shoulder');
                                    const leftWrist = pose.keypoints.find(k => k.name === 'left_wrist');
                                    const rightWrist = pose.keypoints.find(k => k.name === 'right_wrist');
                                    const leftHip = pose.keypoints.find(k => k.name === 'left_hip');
                                    const rightHip = pose.keypoints.find(k => k.name === 'right_hip');

                                    const activeArm = exerciseState.current.currentExerciseType;

                                    let isPoseValid = false;

                                    if (activeArm === 'side_both_arms') {
                                        // Force using left side for left-turn side raise
                                        if (leftShoulder?.score > 0.3 && leftWrist?.score > 0.2) {
                                            isPoseValid = true;
                                        }
                                    } else if (activeArm === 'side_bends') {
                                        // Need shoulders and hips for side bends
                                        if (leftShoulder?.score > 0.3 && rightShoulder?.score > 0.3 &&
                                            leftHip?.score > 0.3 && rightHip?.score > 0.3) {
                                            isPoseValid = true;
                                        }
                                    } else {
                                        // Standard validation for forward-facing exercises
                                        if (leftShoulder?.score > 0.3 && rightShoulder?.score > 0.3 &&
                                            leftWrist?.score > 0.2 && rightWrist?.score > 0.2) {
                                            isPoseValid = true;
                                        }
                                    }

                                    const nose = pose.keypoints.find(k => k.name === 'nose');

                                    // Exercise Logic
                                    if (isPoseValid) {

                                        // Calculate target zones (shoulder height, extended outwards)
                                        const shoulderWidth = Math.abs((leftShoulder?.x || 0) - (rightShoulder?.x || 0)) || 80; // Fallback if side-facing
                                        const armLengthEst = shoulderWidth * 1.5; // Rough estimate

                                        const activeArm = exerciseState.current.currentExerciseType;

                                        let leftTarget, rightTarget;
                                        let midShoulder, midHip, torsoLength, targetBendAngle, currentBendAngleDeg, currentBendDirection;
                                        let headToShoulderDist, targetNeckAngle, currentNeckAngleDeg, currentNeckDirection;

                                        if (activeArm === 'neck_side_bends') {
                                            midShoulder = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
                                            headToShoulderDist = Math.sqrt(Math.pow(midShoulder.x - nose.x, 2) + Math.pow(midShoulder.y - nose.y, 2));
                                            targetNeckAngle = 20 * (Math.PI / 180); // 20 degrees converted to radians

                                            // Calculate current neck angle from vertical (0 is straight up)
                                            const dx_neck = midShoulder.x - nose.x;
                                            const dy_neck = midShoulder.y - nose.y; // Positive since nose is above shoulders
                                            currentNeckAngleDeg = Math.atan2(dx_neck, dy_neck) * (180 / Math.PI);

                                            // For mirrored canvas: >15 is leaning left (screen right), <-15 is leaning right (screen left)
                                            // Since the UI visualizes targetLeftX using positive angle (sin(targetNeckAngle)), 
                                            // visual left actually corresponds to a positive dx (nose is to the right of midShoulder on canvas).
                                            // However, to match the UI targets which are visually left and visually right:
                                            if (currentNeckAngleDeg > 15) currentNeckDirection = 'right'; // visual right
                                            else if (currentNeckAngleDeg < -15) currentNeckDirection = 'left'; // visual left
                                            else currentNeckDirection = 'center';

                                        } else if (activeArm === 'side_bends') {
                                            midShoulder = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
                                            midHip = { x: (leftHip.x + rightHip.x) / 2, y: (leftHip.y + rightHip.y) / 2 };
                                            torsoLength = Math.sqrt(Math.pow(midShoulder.x - midHip.x, 2) + Math.pow(midShoulder.y - midHip.y, 2));
                                            targetBendAngle = 20 * (Math.PI / 180); // 20 degrees converted to radians

                                            // Calculate current angle from vertical (0 is straight up, negative is left, positive is right)
                                            const dx = midShoulder.x - midHip.x;
                                            const dy = midHip.y - midShoulder.y; // Invert Y since canvas Y goes down
                                            currentBendAngleDeg = Math.atan2(dx, dy) * (180 / Math.PI);

                                            // For mirrored canvas: >15 is leaning left (screen right), <-15 is leaning right (screen left)
                                            if (currentBendAngleDeg > 15) currentBendDirection = 'left';
                                            else if (currentBendAngleDeg < -15) currentBendDirection = 'right';
                                            else currentBendDirection = 'center';

                                        } else if (activeArm === 'both_arms') {
                                            // Targets are ABOVE the shoulders for Both Arms Raise (Front Raise)
                                            leftTarget = { x: leftShoulder.x, y: leftShoulder.y - armLengthEst };
                                            rightTarget = { x: rightShoulder.x, y: rightShoulder.y - armLengthEst };
                                        } else if (activeArm === 'side_both_arms') {
                                            // Target is horizontally to the left (same as left arm raise)
                                            const sideArmLength = 120; // Use fixed length since shoulderWidth is compressed
                                            leftTarget = { x: leftShoulder.x + sideArmLength, y: leftShoulder.y };
                                            // rightTarget is not used for this exercise
                                            rightTarget = { x: 0, y: 0 };
                                        } else {
                                            // Targets are TO THE SIDES for Single Arm Raise (Lateral Raise)
                                            leftTarget = { x: leftShoulder.x + armLengthEst, y: leftShoulder.y };
                                            rightTarget = { x: rightShoulder.x - armLengthEst, y: rightShoulder.y };
                                        }
                                        const targetRadius = 40; // Tolerance area

                                        // 1. Draw Target Zones & Guidance Lines
                                        ctx.fillStyle = exerciseState.current.status === 'holding' ? 'rgba(255, 255, 0, 0.5)' : 'rgba(0, 255, 0, 0.3)';
                                        if (exerciseState.current.status === 'completed') ctx.fillStyle = 'rgba(0, 255, 255, 0.5)';

                                        if (activeArm === 'neck_side_bends') {
                                            // Draw neck line (midShoulder to nose)
                                            ctx.beginPath();
                                            ctx.strokeStyle = '#add8e6';
                                            ctx.lineWidth = 4;
                                            ctx.moveTo(midShoulder.x, midShoulder.y);
                                            ctx.lineTo(nose.x, nose.y);
                                            ctx.stroke();

                                            // Draw target angle lines from midShoulder upwards
                                            ctx.beginPath();
                                            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
                                            ctx.setLineDash([5, 5]);

                                            const lineLen = headToShoulderDist * 1.5 || 80;

                                            // Target Left (mirrored)
                                            const targetLeftX = midShoulder.x + lineLen * Math.sin(targetNeckAngle);
                                            const targetLeftY = midShoulder.y - lineLen * Math.cos(targetNeckAngle);
                                            ctx.moveTo(midShoulder.x, midShoulder.y);
                                            ctx.lineTo(targetLeftX, targetLeftY);

                                            // Target Right (mirrored)
                                            const targetRightX = midShoulder.x + lineLen * Math.sin(-targetNeckAngle);
                                            const targetRightY = midShoulder.y - lineLen * Math.cos(-targetNeckAngle);
                                            ctx.moveTo(midShoulder.x, midShoulder.y);
                                            ctx.lineTo(targetRightX, targetRightY);
                                            ctx.stroke();

                                            // Draw target circles
                                            ctx.fillStyle = (exerciseState.current.status === 'holding' && currentNeckDirection === 'left') ? 'rgba(255, 255, 0, 0.6)' : 'rgba(0, 255, 0, 0.3)';
                                            ctx.beginPath();
                                            ctx.arc(targetLeftX, targetLeftY, 15, 0, 2 * Math.PI);
                                            ctx.fill();

                                            ctx.fillStyle = (exerciseState.current.status === 'holding' && currentNeckDirection === 'right') ? 'rgba(255, 255, 0, 0.6)' : 'rgba(0, 255, 0, 0.3)';
                                            ctx.beginPath();
                                            ctx.arc(targetRightX, targetRightY, 15, 0, 2 * Math.PI);
                                            ctx.fill();

                                            ctx.setLineDash([]); // Reset line dash
                                        } else if (activeArm === 'side_bends') {
                                            // Draw spine
                                            ctx.beginPath();
                                            ctx.strokeStyle = 'cyan';
                                            ctx.lineWidth = 4;
                                            ctx.moveTo(midHip.x, midHip.y);
                                            ctx.lineTo(midShoulder.x, midShoulder.y);
                                            ctx.stroke();

                                            // Draw target angle lines from hips
                                            ctx.beginPath();
                                            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
                                            ctx.setLineDash([5, 5]);

                                            const lineLen = torsoLength || 150;
                                            // Target Left
                                            const targetLeftX = midHip.x + lineLen * Math.sin(targetBendAngle);
                                            const targetLeftY = midHip.y - lineLen * Math.cos(targetBendAngle);
                                            ctx.moveTo(midHip.x, midHip.y);
                                            ctx.lineTo(targetLeftX, targetLeftY);

                                            // Target Right
                                            const targetRightX = midHip.x + lineLen * Math.sin(-targetBendAngle);
                                            const targetRightY = midHip.y - lineLen * Math.cos(-targetBendAngle);
                                            ctx.moveTo(midHip.x, midHip.y);
                                            ctx.lineTo(targetRightX, targetRightY);
                                            ctx.stroke();

                                            // Draw target circles at the ends of the lines
                                            ctx.fillStyle = (exerciseState.current.status === 'holding' && currentBendDirection === 'left') ? 'rgba(255, 255, 0, 0.6)' : 'rgba(0, 255, 0, 0.3)';
                                            ctx.beginPath();
                                            ctx.arc(targetLeftX, targetLeftY, 20, 0, 2 * Math.PI);
                                            ctx.fill();

                                            ctx.fillStyle = (exerciseState.current.status === 'holding' && currentBendDirection === 'right') ? 'rgba(255, 255, 0, 0.6)' : 'rgba(0, 255, 0, 0.3)';
                                            ctx.beginPath();
                                            ctx.arc(targetRightX, targetRightY, 20, 0, 2 * Math.PI);
                                            ctx.fill();

                                            ctx.setLineDash([]); // Reset line dash
                                        } else {
                                            if (activeArm === 'left_arm' || activeArm === 'both_arms' || activeArm === 'side_both_arms') {
                                                // Left target
                                                ctx.beginPath();
                                                ctx.arc(leftTarget.x, leftTarget.y, targetRadius, 0, 2 * Math.PI);
                                                ctx.fill();
                                            }
                                            if (activeArm === 'right_arm' || activeArm === 'both_arms') {
                                                // Right target
                                                ctx.beginPath();
                                                ctx.arc(rightTarget.x, rightTarget.y, targetRadius, 0, 2 * Math.PI);
                                                ctx.fill();
                                            }

                                            // 2. Draw Guidance Lines
                                            ctx.beginPath();
                                            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
                                            ctx.lineWidth = 2;
                                            ctx.setLineDash([5, 5]);

                                            if (activeArm === 'left_arm' || activeArm === 'both_arms' || activeArm === 'side_both_arms') {
                                                // Line from left wrist to left target
                                                ctx.moveTo(leftWrist.x, leftWrist.y);
                                                ctx.lineTo(leftTarget.x, leftTarget.y);
                                            }
                                            if (activeArm === 'right_arm' || activeArm === 'both_arms') {
                                                // Line from right wrist to right target
                                                ctx.moveTo(rightWrist.x, rightWrist.y);
                                                ctx.lineTo(rightTarget.x, rightTarget.y);
                                            }
                                            ctx.stroke();
                                            ctx.setLineDash([]); // Reset line dash
                                        }

                                        // 3. Evaluate Position (Single Arm & Both Arms)
                                        let isArmRaised = false;
                                        let isArmLowered = false;
                                        let distanceY = 0; // for debug display

                                        const verticalTolerance = 60; // pixels
                                        const lowerThreshold = 100;

                                        if (activeArm === 'neck_side_bends') {
                                            // Neck Side Bends Evaluation
                                            const requiredAngle = 18; // degrees
                                            distanceY = currentNeckAngleDeg; // visual debug

                                            isArmLowered = currentNeckDirection === 'center'; // "lowered" means returned to center

                                            // Check if leaning head enough, and NOT the same direction
                                            if ((currentNeckDirection === 'left' || currentNeckDirection === 'right') &&
                                                Math.abs(currentNeckAngleDeg) > requiredAngle) {

                                                if (exerciseState.current.lastSideBendDirection !== currentNeckDirection) {
                                                    isArmRaised = true; // "raised" means leaning head correctly
                                                } else {
                                                    // Prevent holding the same side multiple times
                                                    if (exerciseState.current.status !== 'holding') {
                                                        setInstruction(`Tilt to the OTHER side!`);
                                                    }
                                                }
                                            }
                                        } else if (activeArm === 'side_bends') {
                                            // Side Bends Evaluation
                                            const requiredAngle = 18; // degrees
                                            distanceY = currentBendAngleDeg; // visual debug

                                            isArmLowered = currentBendDirection === 'center'; // "lowered" means returned to center

                                            // Check if leaning enough, and NOT the same direction as the immediate last successful rep
                                            if ((currentBendDirection === 'left' || currentBendDirection === 'right') &&
                                                Math.abs(currentBendAngleDeg) > requiredAngle) {

                                                // Either we haven't done any reps yet, or we are alternating
                                                if (exerciseState.current.lastSideBendDirection !== currentBendDirection) {
                                                    isArmRaised = true; // "raised" means leaning correctly
                                                } else {
                                                    // Prevent holding the same side multiple times
                                                    if (exerciseState.current.status !== 'holding') {
                                                        setInstruction(`Lean to the OTHER side!`);
                                                    }
                                                }
                                            }
                                        } else if (activeArm === 'side_both_arms') {
                                            // Side Both Arms Raise (fixed to left side)
                                            // Check if wrist is at shoulder height
                                            distanceY = Math.abs(leftWrist.y - leftShoulder.y);

                                            isArmRaised = distanceY < verticalTolerance;
                                            isArmLowered = leftWrist.y > leftShoulder.y + lowerThreshold;

                                        } else if (activeArm === 'both_arms') {
                                            // Both Arms Raise: check if both wrists are high up and not too far out
                                            // Y distance to target (should be close to Target)
                                            const leftDistY = Math.abs(leftWrist.y - leftTarget.y);
                                            const rightDistY = Math.abs(rightWrist.y - rightTarget.y);

                                            // X deviation from shoulders (prevents doing a lateral raise instead)
                                            const leftDeviationX = Math.abs(leftWrist.x - leftShoulder.x);
                                            const rightDeviationX = Math.abs(rightWrist.x - rightShoulder.x);
                                            // Allow some horizontal spread but enforce bringing arms 'forward/up'
                                            const horizontalTolerance = shoulderWidth * 0.8;

                                            isArmRaised = leftDistY < verticalTolerance && rightDistY < verticalTolerance &&
                                                leftDeviationX < horizontalTolerance && rightDeviationX < horizontalTolerance;

                                            // Lowered when wrists are down below shoulders again
                                            isArmLowered = leftWrist.y > leftShoulder.y + lowerThreshold && rightWrist.y > rightShoulder.y + lowerThreshold;

                                            distanceY = (leftDistY + rightDistY) / 2; // Average for debug

                                        } else {
                                            // Single Arm Lateral Raise
                                            const leftDistanceY = Math.abs(leftWrist.y - leftShoulder.y);
                                            const rightDistanceY = Math.abs(rightWrist.y - rightShoulder.y);

                                            distanceY = activeArm === 'left_arm' ? leftDistanceY : rightDistanceY;

                                            isArmRaised = distanceY < verticalTolerance;

                                            isArmLowered = activeArm === 'left_arm'
                                                ? leftWrist.y > leftShoulder.y + lowerThreshold
                                                : rightWrist.y > rightShoulder.y + lowerThreshold;
                                        }

                                        // --- Debugging output directly on canvas ---
                                        ctx.fillStyle = 'white';
                                        ctx.font = '16px Arial';
                                        ctx.fillText(`Target: ${activeArm === 'left_arm' ? 'Left' : 'Right'} | Dist/Ang: ${Math.round(distanceY)}`, 10, 30);
                                        ctx.fillText(`State: ${exerciseState.current.status} | Raised: ${isArmRaised}`, 10, 50);
                                        // ------------------------------------------

                                        const state = exerciseState.current;
                                        const now = Date.now();

                                        // Do not process if both are completed (wait for a full reset eventually)
                                        if (state.status === 'all_completed') {
                                            ctx.fillText(`All Exercises Completed!`, 10, 70);
                                            return;
                                        }

                                        if (state.status === 'resting' && isArmRaised) {
                                            // Start holding
                                            state.status = 'holding';
                                            state.holdStartTime = now;
                                            state.graceFrames = 0;
                                            console.log("Started holding!");
                                        } else if (state.status === 'holding') {
                                            if (isArmRaised) {
                                                state.graceFrames = 0;
                                                const holdDuration = now - state.holdStartTime;
                                                const remainingTime = Math.max(0, Math.ceil((state.targetHoldTime - holdDuration) / 1000));

                                                if (holdDuration >= state.targetHoldTime) {
                                                    // Rep completed
                                                    state.status = 'completed';
                                                    state.reps += 1;

                                                    if (state.reps >= 8 && state.currentExerciseType === 'right_arm') {
                                                        // === AI Social Agent: Triggered after completing a set ===
                                                        triggerAIFeedback('right_arm', mockUserData, showAiMessage, hideAiMessage);
                                                        // Switch to left arm
                                                        state.currentExerciseType = 'left_arm';
                                                        state.reps = 0;
                                                        setCurrentExerciseTracker('Left Arm Raise');
                                                        setInstruction('Switch to Left Arm! Raise to shoulder height');
                                                    } else if (state.reps >= 8 && state.currentExerciseType === 'left_arm') {
                                                        // === AI Social Agent: Triggered after completing a set ===
                                                        triggerAIFeedback('left_arm', mockUserData, showAiMessage, hideAiMessage);
                                                        // Switch to both arms
                                                        state.currentExerciseType = 'both_arms';
                                                        state.reps = 0;
                                                        setCurrentExerciseTracker('Both Arms Raise');
                                                        setInstruction('Switch to Both Arms! Raise forwards over head');
                                                    } else if (state.reps >= 8 && state.currentExerciseType === 'both_arms') {
                                                        // === AI Social Agent: Triggered after completing a set ===
                                                        triggerAIFeedback('both_arms', mockUserData, showAiMessage, hideAiMessage);
                                                        // Switch to side both arms
                                                        state.currentExerciseType = 'side_both_arms';
                                                        state.reps = 0;
                                                        setCurrentExerciseTracker('Side Both Arms Raise');
                                                        setInstruction('Turn Left! Raise arms forward');
                                                    } else if (state.reps >= 8 && state.currentExerciseType === 'side_both_arms') {
                                                        // Switch to side bends
                                                        state.currentExerciseType = 'side_bends';
                                                        state.reps = 0;
                                                        state.lastSideBendDirection = null;
                                                        setCurrentExerciseTracker('Side Bends');
                                                        setInstruction('Hands on hips! Lean to one side');
                                                    } else if (state.reps >= 8 && state.currentExerciseType === 'side_bends') {
                                                        // Switch to neck side bends
                                                        state.currentExerciseType = 'neck_side_bends';
                                                        state.reps = 0;
                                                        state.lastSideBendDirection = null;
                                                        setCurrentExerciseTracker('Neck Side Bends');
                                                        setInstruction('Keep shoulders still! Tilt your head to one side');
                                                    } else if (state.reps >= 8 && state.currentExerciseType === 'neck_side_bends') {
                                                        state.status = 'all_completed';
                                                        setInstruction('All Exercises Completed!');
                                                    } else {
                                                        if (activeArm === 'side_bends' || activeArm === 'neck_side_bends') {
                                                            // Use the matching direction variable
                                                            const curDir = activeArm === 'side_bends' ? currentBendDirection : currentNeckDirection;
                                                            state.lastSideBendDirection = curDir; // Remember the side just completed
                                                            setInstruction('Great! Return to center then lean to the OTHER side.');
                                                        } else {
                                                            setInstruction('Great! Lower arm(s)');
                                                        }
                                                    }

                                                    setReps(state.reps);
                                                    console.log("Rep completed!");
                                                } else {
                                                    // Update UI with countdown
                                                    setInstruction(`Hold... ${remainingTime}s`);

                                                    // Draw visual countdown circle
                                                    const progress = holdDuration / state.targetHoldTime;
                                                    ctx.beginPath();
                                                    ctx.strokeStyle = '#00FF00';
                                                    ctx.lineWidth = 5;

                                                    if (activeArm === 'neck_side_bends') {
                                                        const lineLen = headToShoulderDist * 1.5 || 80;
                                                        let cx, cy;
                                                        if (currentNeckDirection === 'left') {
                                                            cx = midShoulder.x + lineLen * Math.sin(targetNeckAngle);
                                                            cy = midShoulder.y - lineLen * Math.cos(targetNeckAngle);
                                                        } else {
                                                            cx = midShoulder.x + lineLen * Math.sin(-targetNeckAngle);
                                                            cy = midShoulder.y - lineLen * Math.cos(-targetNeckAngle);
                                                        }
                                                        ctx.arc(cx, cy, 25, -Math.PI / 2, (-Math.PI / 2) + (2 * Math.PI * progress));
                                                        ctx.stroke();

                                                    } else if (activeArm === 'side_bends') {
                                                        const lineLen = torsoLength || 150;
                                                        let cx, cy;
                                                        if (currentBendDirection === 'left') {
                                                            cx = midHip.x + lineLen * Math.sin(targetBendAngle);
                                                            cy = midHip.y - lineLen * Math.cos(targetBendAngle);
                                                        } else {
                                                            cx = midHip.x + lineLen * Math.sin(-targetBendAngle);
                                                            cy = midHip.y - lineLen * Math.cos(-targetBendAngle);
                                                        }
                                                        ctx.arc(cx, cy, 25, -Math.PI / 2, (-Math.PI / 2) + (2 * Math.PI * progress));
                                                        ctx.stroke();

                                                    } else {
                                                        if (activeArm === 'left_arm' || activeArm === 'both_arms' || activeArm === 'side_both_arms') {
                                                            ctx.arc(leftTarget.x, leftTarget.y, targetRadius + 5, -Math.PI / 2, (-Math.PI / 2) + (2 * Math.PI * progress));
                                                        }
                                                        ctx.stroke();

                                                        if (activeArm === 'both_arms') {
                                                            ctx.beginPath();
                                                            ctx.arc(rightTarget.x, rightTarget.y, targetRadius + 5, -Math.PI / 2, (-Math.PI / 2) + (2 * Math.PI * progress));
                                                            ctx.stroke();
                                                        } else if (activeArm === 'right_arm') {
                                                            ctx.beginPath();
                                                            ctx.arc(rightTarget.x, rightTarget.y, targetRadius + 5, -Math.PI / 2, (-Math.PI / 2) + (2 * Math.PI * progress));
                                                            ctx.stroke();
                                                        }
                                                    }
                                                }
                                            } else {
                                                // Tolerate sporadic tracking loss
                                                state.graceFrames += 1;
                                                if (state.graceFrames > 15) { // roughly 0.5 sec at 30 fps
                                                    state.status = 'resting';
                                                    let instructionTxt = 'Raise right arm to shoulder height';
                                                    if (activeArm === 'left_arm') instructionTxt = 'Raise left arm to shoulder height';
                                                    if (activeArm === 'both_arms') instructionTxt = 'Raise both arms forwards over head';
                                                    if (activeArm === 'side_both_arms') instructionTxt = 'Turn Left! Raise arms forward';
                                                    if (activeArm === 'side_bends') instructionTxt = state.lastSideBendDirection ? 'Return to center then lean to the OTHER side.' : 'Hands on hips! Lean to one side';
                                                    if (activeArm === 'neck_side_bends') instructionTxt = state.lastSideBendDirection ? 'Return to center then head tilt to the OTHER side.' : 'Keep shoulders still! Tilt your head to one side';
                                                    setInstruction(instructionTxt);
                                                    console.log("Dropped early, holding cancelled.");
                                                }
                                            }
                                        } else if (state.status === 'completed' && isArmLowered) {
                                            // Reset for next rep
                                            state.status = 'resting';
                                            let instructionTxt = 'Raise right arm to shoulder height';
                                            if (activeArm === 'left_arm') instructionTxt = 'Raise left arm to shoulder height';
                                            if (activeArm === 'both_arms') instructionTxt = 'Raise both arms forwards over head';
                                            if (activeArm === 'side_both_arms') instructionTxt = 'Turn Left! Raise arms forward';
                                            if (activeArm === 'side_bends') instructionTxt = state.lastSideBendDirection ? 'Lean to the OTHER side.' : 'Hands on hips! Lean to one side';
                                            if (activeArm === 'neck_side_bends') instructionTxt = state.lastSideBendDirection ? 'Tilt head to the OTHER side.' : 'Keep shoulders still! Tilt your head to one side';
                                            setInstruction(instructionTxt);
                                            console.log("Arms/Torso returned, ready for next rep.");
                                        }
                                    }
                                });
                            }
                        } catch (error) {
                            console.error("Error during pose estimation:", error);
                        }
                    }

                    requestAnimationFrame(detect);
                };

                detect();
            } catch (err) {
                console.error("Initialization error:", err);
            }
        };

        runPoseDetection();
    }, [showAiMessage, hideAiMessage]);

    return (
        <div className="App">
            <header className="App-header">
                <h1>RehabiMate</h1>
                {loading && <p>Loading TensorFlow model...</p>}
                <div className="camera-container">
                    <div className="ui-overlay">
                        <div className="rep-counter">{currentExerciseTracker}: {reps}/8</div>
                        <div className="instruction-text">{instruction}</div>
                    </div>
                    {aiMessage && (
                        <div className="ai-feedback">
                            <span className="ai-feedback-icon">🤖</span>
                            <span className="ai-feedback-text">{aiMessage}</span>
                        </div>
                    )}
                    <video
                        ref={videoRef}
                        style={{
                            position: 'absolute',
                            marginLeft: 'auto',
                            marginRight: 'auto',
                            left: 0,
                            right: 0,
                            textAlign: 'center',
                            zIndex: 9,
                            width: 360,
                            height: 640,
                            transform: 'scaleX(-1)',
                        }}
                    />
                    <canvas
                        ref={canvasRef}
                        style={{
                            position: 'absolute',
                            marginLeft: 'auto',
                            marginRight: 'auto',
                            left: 0,
                            right: 0,
                            textAlign: 'center',
                            zIndex: 10,
                            width: 360,
                            height: 640,
                            transform: 'scaleX(-1)',
                        }}
                    />
                </div>
            </header>
        </div>
    );
}

export default App;