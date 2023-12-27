from fastapi import FastAPI, File, UploadFile,Query
from fastapi.responses import JSONResponse
from pipeline import ExtractionAndVectorStoringTask,ResponseTask
from config import PDF_DATA_PATH
import os


app = FastAPI()

@app.get("/")
def read_root():
    print("hellll")
    return {"Hello": "World"}


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Check if the file is a PDF
    if file.filename.endswith('.pdf'):
        file_location = os.path.join(PDF_DATA_PATH, file.filename)
        
        # Save the uploaded PDF to the specified folder
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())

        vector_store = ExtractionAndVectorStoringTask(folder_path=PDF_DATA_PATH)
        vector_store.vector_storing()
        
        return JSONResponse(content={"message": "PDF uploaded successfully"}, status_code=200)
    else:
        return JSONResponse(content={"error": "Invalid file format. Please upload a PDF."}, status_code=400)

@app.get("/response/")
def read_root(queation: str = Query(..., description="Your queation")):
    qa = ResponseTask(query=queation)
    
    return {"message": qa.get_response()}

if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:8000/docs")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)