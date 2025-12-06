import axios from "axios";
import { API_URL } from "../constant";

const loginRequest = async (username: string, password: string) => {
  const response = await axios.post(
    `${API_URL}/auth/login/`,
    { username, password },
  );
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to login");
  }
  return response.data;
};

const registerRequest = async (username: string, email: string, password: string) => {
  const response = await axios.post(
    `${API_URL}/auth/register/`,
    { username, email, password },
  );
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to register");
  }
};

const postPrompt = async (prompt: string) => {
  const response = await axios.post(
    `${API_URL}/core/prompt/`,
    { prompt },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to post prompt");
  }
};

const postProofreader = async (prompt: string) => {
  const response = await axios.post(
    `${API_URL}/grammar/proofreader/`,
    { prompt },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to post proofreader");
  }
};

const postSummarizer = async (prompt: string) => {
  const response = await axios.post(
    `${API_URL}/document/summarizer/`,
    { prompt },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to post summarizer");
  }
};

const postTranslator = async (
  prompt: string,
  targetLanguage: string,
  sourceLanguage: string
) => {
  const response = await axios.post(
    `${API_URL}/grammar/translator/`,
    {
      prompt,
      target_language: targetLanguage,
      source_language: sourceLanguage,
    },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to post translator");
  }
};

const postWriter = async (prompt: string) => {
  const response = await axios.post(
    `${API_URL}/document/writer/`,
    { prompt },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to post writer");
  }
};

const postRewriter = async (prompt: string) => {
  const response = await axios.post(
    `${API_URL}/document/rewriter/`,
    { prompt },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to post rewriter");
  }
};

const postCopywriting = async (prompt: string) => {
  const response = await axios.post(
    `${API_URL}/grammar/copywriting/`,
    { prompt },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );

  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to post copywriting");
  }
};

const postExplainer = async (prompt: string) => {
  const response = await axios.post(
    `${API_URL}/core/explainer/`,
    { prompt },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );

  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to post explainer");
  }
};

const getHistory = async () => {
  const response = await axios.get(`${API_URL}/core/history/`, {
    headers: {
      Authorization: localStorage.getItem("apiKey"),
    },
  });
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to get history");
  }
};

const handleResponseData = (response: string) => {
  if (typeof response === "string" && response.charAt(0) === "{") {
    const parsedData = JSON.parse(response);
    if (parsedData.response) {
      return parsedData.response;
    }
  }
  return response;
};

const insertFile = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await axios.post(`${API_URL}/document/pdf-upload/`, formData, {
    headers: {
      Authorization: localStorage.getItem("apiKey"),
    },
  });
  if (response.status === 200) {
    console.log("File inserted successfully");
    return response.data;
  } else {
    console.log("Failed to insert file");
    throw new Error("Failed to insert file");
  }
};

const chatWithRAG = async (prompt: string) => {
  const response = await axios.post(
    `${API_URL}/document/rag-chat/`,
    { prompt },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to insert file");
  }
};

const generateImage = async (prompt: string) => {
  const response = await axios.post(
    `${API_URL}/core/image/`,
    { prompt },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );

  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to generate image");
  }
};

const createEmail = async (
  context: string,
  recipients: string,
  sender: string,
  prompt: string
) => {
  const response = await axios.post(
    `${API_URL}/document/email-generator/`,
    { context, recipients, sender, prompt },
    {
      headers: {
        Authorization: localStorage.getItem("apiKey"),
      },
    }
  );
  if (response.status === 200) {
    return response.data;
  } else {
    throw new Error("Failed to create email");
  }
};

const services = {
  loginRequest,
  registerRequest,
  postPrompt,
  postProofreader,
  postSummarizer,
  postTranslator,
  postWriter,
  postRewriter,
  postCopywriting,
  postExplainer,
  getHistory,
  chatWithRAG,
  insertFile,
  handleResponseData,
  generateImage,
  createEmail,
};

export default services;