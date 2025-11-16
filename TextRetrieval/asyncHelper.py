import asyncio
import nest_asyncio


async def _await_in_background(coro):
    """Actually await the coroutine so LangChain can handle it."""
    return await coro

def run_async(coro):
    """Run async code safely from sync (StructuredTool) calls."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Running loop (Jupyter/IPython) → apply nest_asyncio
    
    nest_asyncio.apply()

    return loop.run_until_complete(coro)