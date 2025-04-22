"""
Application to run the whole program
"""
import libs.LanguageProcessing as lp
import libs.API_ImageRequestHandler as API_handler
import libs.StreetArtTextExtractor as StreetArtTextExtractor


if __name__ == "__main__":
    """
    Main function for the total application
    """
    # Mappilary access token
    TOKEN = 'MLY|9398391603544096|d47348ca942d8d06c2d47825aa4c6f70'
    #Initialize API request class
    fetcher = API_handler.MapillaryImageFetcher(TOKEN)
    #Initialize language processing class
    language_processor = lp.LanguageProcessing()
    # Initialize text extractor module
    text_extractor = StreetArtTextExtractor.StreetArtTextExtractor()
    
    city_information={
        # "Example Entry":{
        #     "longitude": ...,
        #     "latitude": ...,
        #     "info":{"total_images":0,"profanity":0,"clear_images":0}
        # },
        "Athens (Exarcheia)":{
            "longitude": 23.7314,
            "latitude": 37.9840,
            "info":{"total_images":0,"profanity":0,"clear_images":0}
        },
        "Naples (Centro Storico)":{
            "longitude": 14.2681,
            "latitude": 40.8520,
            "info":{"total_images":0,"profanity":0,"clear_images":0}
        },
        "Santiago (Barrio Yungay)":{
            "longitude": -70.6780,
            "latitude": -33.4432,
            "info":{"total_images":0,"profanity":0,"clear_images":0}
        },
        "Baltimore (West Baltimore)":{
            "longitude": -76.6413,
            "latitude": 39.2952,
            "info":{"total_images":0,"profanity":0,"clear_images":0}
        },
        "Marseille (Noailles)":{
            "longitude": 5.3811,
            "latitude": 43.2951,
            "info":{"total_images":0,"profanity":0,"clear_images":0}
        },
        "Detroit (Delray)":{
            "longitude": -83.0458,
            "latitude": 42.3314,
            "info":{"total_images":0,"profanity":0,"clear_images":0}
        },
        "Rio de Janeiro (Lapa)":{
            "longitude": -43.1729,
            "latitude": -22.9068,
            "info":{"total_images":0,"profanity":0,"clear_images":0}
        },
        "Tijuana (Zona Norte)":{
            "longitude": -117.0382,
            "latitude": 32.5149,
            "info":{"total_images":0,"profanity":0,"clear_images":0}
        },
        "Barcelona (El Raval)":{
            "longitude": 2.1734,
            "latitude": 41.3851,
            "info":{"total_images":0,"profanity":0,"clear_images":0}
        },
        "Johannesburg (Yeoville)":{
            "longitude": 28.0476,
            "latitude": -26.2041,
            "info":{"total_images":0,"profanity":0,"clear_images":0}
        }
    }
     
    for city,data in city_information.items():
        print(f"Fetching images for {city}...")
        latitude = data["latitude"]
        longitude = data["longitude"]

        print(f"Fetching images...")
        image_list = fetcher.fetch_and_process_images(latitude, longitude, radius=500)

        #Extract text from the ret5rieved batch of images
        print(f"Extracting text...")
        retrieved_text = text_extractor.process_batch_of_images(image_list,preprocessing="contrast")

        # Do the language Processing and the profanity check
        print(f"Processing text found...")
        profanity_count,clear_count = language_processor.do_profanity_check(retrieved_text)

        # Store results 
        #city_information [city]["data"]["total_images"] += len(rerieved_text)
        print(f"Saving metrics")
        city_information[city]["info"]["profanity"] += profanity_count
        city_information[city]["info"]["clear_images"] += clear_count


    #Print final results
    for city,data in city_information.items():
        print(f"For city:{city} detected profanity : {city_information[city]['info']['profanity']}, clear_images:{city_information[city]['info']['clear_images']}")
