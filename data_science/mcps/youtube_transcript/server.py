from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field
from youtube_transcript_api import YouTubeTranscriptApiException

from .services import YouTubeTranscriptService

mcp = MCPServer(
    name="YouTube Transcript",
    instructions="Get YouTube transcript of a video as plain text.",
)

_service = YouTubeTranscriptService(use_proxy=True)


@mcp.tool(
    name="get_youtube_transcript",
    description="Get YouTube transcript of a video as plain text.",
)
def get_youtube_transcript(
    video_url_or_id: Annotated[
        str, Field(description="The URL or ID of the YouTube video.")
    ],
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
        return _service.get_transcript_text(video_url_or_id)
    except YouTubeTranscriptApiException as error:
        return f"Error: {error!s}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
