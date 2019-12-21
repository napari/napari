import weakref
import zprocess
from qtutils import inmain_decorator


class RemoteServer(zprocess.ZMQServer):
    def __init__(self, parent, *args, **kwargs):
        self.parent = weakref.ref(parent)
        super().__init__(*args, **kwargs)

    @inmain_decorator()
    def handler(self, msg):
        """Description.

        Arguments:
            msg {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        if isinstance(msg, (tuple, list)) and len(msg) == 3:
            name, args, kwargs = msg
            # Is there an explicit handler for the request?
            if hasattr(self, 'handle_' + name):
                return getattr(self, 'handle_' + name)(*args, **kwargs)
            # Run the parent method if it exists
            if hasattr(self.parent(), name):
                fn = getattr(self.parent(), name)
                if callable(fn):
                    # This is a callable method, so call it
                    fn(*args, **kwargs)
                    return True
                else:
                    # This is an attribute/property, so return it
                    return fn
        elif msg == 'hello':
            return 'hello'
        return "error: request not supported."

    def handle_close(self):
        """Close the napari window (equivalent to Ctrl+W)."""
        self.parent().window.close()
        return True

    def handle_hello(self):
        """Simple req/rep confirmation."""
        return "hello"

    def handle_get_data(self, layer=0):
        """Get image data from the specified layer."""
        return self.parent().layers[layer].data


class Client(zprocess.ZMQClient):
    """A zprocess.ZMQClient for communicating with napari."""

    def __init__(self, host='localhost', port=7341, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        super().__init__()

    def request(self, name, *args, **kwargs):
        """Call a method of napari.viewer.Viewer remotely or get
        an instance attribute/property. If `name` is a method, args
        and kwargs are passed.
        """
        return self.get(
            port=self.port,
            host=self.host,
            data=[name, args, kwargs],
            timeout=self.timeout,
        )

    def say_hello(self):
        """Ping the napari server for a response."""
        return self.request('hello')

    def get_version(self):
        """Return the version of napari the server is running in."""
        return self.request('version')

    def reset_view(self):
        """Reset the view (equivalent to Ctrl+R)."""
        return self.request('reset_view')
