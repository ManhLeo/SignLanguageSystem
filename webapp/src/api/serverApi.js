import axios from 'axios';

const API_URL = 'http://127.0.0.2:5000/api'; // Thay đổi URL tùy theo server của bạn

export const uploadImage = async (imageData) => {
    try {
        const response = await axios.post(`${API_URL}/recognize`, {
            image: imageData
        });
        return response.data;
    } catch (error) {
        console.error('Error uploading image:', error);
        throw error;
    }
};

export const updateRecognitionResult = async (result) => {
    try {
        const response = await axios.post(`${API_URL}/update-result`, result);
        return response.data;
    } catch (error) {
        console.error('Error updating result:', error);
        throw error;
    }
};

export const getLatestResult = async () => {
    try {
        const response = await axios.get(`${API_URL}/latest-result`);
        return response.data;
    } catch (error) {
        console.error('Error getting latest result:', error);
        throw error;
    }
}; 