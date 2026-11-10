"""Errores del pipeline, con el código del catálogo (``IA_*``, API_IA_BACK §2.5)."""


class PipelineError(Exception):
    """Error con código estable. La API lo traduce a HTTP y el escritorio a un mensaje."""

    def __init__(self, code: str, message: str = "", **context):
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.context = context


class PipelineCancelled(Exception):
    """El llamador pidió detener el procesamiento (``should_stop`` devolvió True)."""
