from typing import Annotated

from mcp.server.fastmcp import FastMCP
from services import YouTubeTranscriptService

mcp = FastMCP(
    name="YouTube Transcript",
    description="Get YouTube transcript of a video as plain text.",
    stateless_http=True,
    annotation="This tool fetches the transcript of a YouTube video given its URL or ID. It returns the transcript as plain text.",
)

_service = YouTubeTranscriptService(use_proxy=True)


@mcp.tool(
    name="get_youtube_transcript",
    description="Get YouTube transcript of a video as plain text.",
    parameters={
        "video_url_or_id": {
            "type": "string",
            "description": "The URL or ID of the YouTube video.",
        }
    },
)
def get_youtube_transcript(
    video_url_or_id: Annotated[str, "The URL or ID of the YouTube video."],
) -> str:
    """Get YouTube transcript of a video as plain text.

    Parameters
    ----------
    video_url_or_id : str
        The URL or ID of the YouTube video.

    Returns
    -------
    str
        The transcript of the video as plain text.
    """

    try:
        return _service.get_transcript(video_url_or_id)
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
