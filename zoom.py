import asyncio
from playwright.async_api import async_playwright
import time
names = []



async def join_zoom_meeting(context, semaphore, name):
    # Limit concurrency per browser session with semaphore
    async with semaphore:
        try:
            # Open a new page within the context
            page = await context.new_page()

            while True:  # Retry loop for handling the "Retry" button
                try:
                    # Wait for the Zoom page to load and for the iframe to appear
                    await page.goto(
                        "https://app.zoom.us/wc/94825910937/join?fromPWA=1&pwd=5FoFcEzbdYMthZ4PbtK6AfO7IXaMuY.1&_x_zm_rtaid=6HHFC_dcR8ysX7hgPT1AGA.1737028335642.c61bef5e9aaf116e5ba5e2018b7581d6&_x_zm_rhtaid=391"                      )
                    await page.wait_for_selector('iframe.pwa-webclient__iframe')

                    # Get the iframe element
                    iframe_element = await page.query_selector('iframe.pwa-webclient__iframe')
                    iframe = await iframe_element.content_frame()  # Get the content frame from the iframe

                    # Locate and click the "Accept Cookies" button if it exists
                    accept_button = iframe.locator('#onetrust-accept-btn-handler')
                    if await accept_button.is_visible():
                        await accept_button.click()
                        print("Clicked 'Accept Cookies' button.")

                    # Wait for the name input field to appear inside the iframe
                    await iframe.wait_for_selector('input#input-for-name', timeout=9999999)

                    # Type "name" into the input field
                    name_input = await iframe.query_selector('input#input-for-name')
                    if name_input:
                        await name_input.fill(name)  # Fill the input with the string "name"

                    # Find the "Join" button and click it
                    join_button = iframe.locator(
                        'button.zm-btn.preview-join-button.zm-btn--default.zm-btn__outline--blue'
                    )
                    await join_button.click()
                    print("Attempted to join the meeting.")

                    # Wait to see if the "Retry" button appears
                    retry_button = page.locator(
                        'button.zmu-btn.zm-btn-legacy.zmu-btn--primary.zmu-btn__outline--blue'
                    )
                    if await retry_button.is_visible():
                        print("Retry button detected. Restarting the join process...")
                        await retry_button.click()  # Click the Retry button
                        await asyncio.sleep(2)  # Short delay before retrying
                        continue  # Restart the process

                    print(name," Joined the meeting successfully.")
                    break  # Exit the retry loop if no "Retry" button is found

                except Exception as e:
                    print(f"Error occurred during join attempt: {e}")
                    break  # Exit the loop on error to avoid infinite retries

            # Wait indefinitely for the meeting to continue
            await asyncio.Event().wait()  # Keeps the browser open

        except Exception as e:
            print(f"Error occurred: {e}")

async def run_zoom_instances(onebyone=False):
    names = input("Enter names separated by comma : ").replace("\"","").split(",")
    print(names)
    async with async_playwright() as p:
        # Launch the browser with default arguments
        browser = await p.chromium.launch(headless=False)

        # Limit concurrency with a semaphore
        semaphore = asyncio.Semaphore(50)

        tasks = []
        j = 0
        for i in names:  # Number of Zoom instances to join
            j+=1
            time.sleep(1)
            
            # Create a new browser context with microphone permissions
            if(onebyone):
                i = input("Enter name : ")
            context = await browser.new_context(
                
                permissions=["microphone"],  # Grant microphone permissions
                viewport={"width": 480, "height": 360},  # Optional: Adjust viewport size
            )
            await context.grant_permissions(["microphone"])  # Grant microphone permissions just in case

            # Add the task for joining a Zoom meeting
            tasks.append(asyncio.create_task(join_zoom_meeting(context, semaphore,i)))

        # Wait for all tasks to finish
        await asyncio.gather(*tasks)

        # Close the browser instance after all tasks are completed
        await browser.close()

# Start the main loop
if __name__ == "__main__":

    asyncio.run(run_zoom_instances())
